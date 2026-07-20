"""Hybrid A* path planner for a skid‑steer robot pulling a trailer.

The planner expands nodes using the `Simulator.step` method, which integrates the
kinematic model of the robot‑trailer system.  It receives a start and goal
`RobotState` and returns a list of states representing a feasible path.

This implementation is deliberately lightweight and intended as a reference
implementation that can be extended (e.g., adding motion‑primitive pruning,
collision checking, or a more sophisticated heuristic).
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Tuple

from simulator import RobotState, Simulator
import numpy as np
import math
from numba import njit

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _heuristic(state: RobotState, goal: RobotState) -> float:
    """Euclidean distance between the robot positions (ignoring orientation).

    A simple admissible heuristic for the hybrid A* algorithm.  More elaborate
    heuristics could incorporate the trailer geometry, but this works for a
    basic demonstration.
    """
    dx = state.x - goal.x
    dy = state.y - goal.y
    return math.hypot(dx, dy)


def _discretize(state: RobotState, pos_res: float = 0.1, ang_res: float = math.radians(5)) -> Tuple[int, int, int, int]:
    """Return a hashable discretized representation of a state.

    The continuous state space is quantised to enable a closed‑set lookup.
    """
    x_i = int(round(state.x / pos_res))
    y_i = int(round(state.y / pos_res))
    theta_i = int(round(state.theta_robot / ang_res))
    beta_i = int(round(state.beta / ang_res))
    return (x_i, y_i, theta_i, beta_i)


# ---------------------------------------------------------------------------
# Planner class
# ---------------------------------------------------------------------------

class HybridAStarPlanner:
    """Hybrid A* planner for the robot‑trailer system with obstacle avoidance.

    Parameters
    ----------
    simulator: Simulator
        Kinematic simulator used to propagate the dynamics.
    control_set: List[Tuple[float, float]]
        Motion primitives expressed as left/right wheel velocities.
    occupancy_grid: np.ndarray | None, optional
        2‑D array where ``True``/``1`` indicates an occupied cell. ``None``
        disables collision checking (default).
    grid_resolution: float, optional
        Size of a grid cell in world metres. Required when an occupancy grid is
        supplied. Defaults to ``0.1`` m.
    grid_origin: Tuple[float, float], optional
        World coordinates of the grid cell ``(0, 0)`` (lower‑left corner).
        Defaults to ``(0.0, 0.0)``.
    dt: float, optional
        Integration time step for each expansion (default matches visualiser).
    """

    def __init__(
        self,
        simulator: Simulator,
        control_set: List[Tuple[float, float]],
        occupancy_grid: np.ndarray | None = None,
        grid_resolution: float = 0.1,
        grid_origin: Tuple[float, float] = (0.0, 0.0),
        dt: float = 0.1,
    ):
        self.simulator = simulator
        self.control_set = control_set
        self.dt = dt
        self.occupancy_grid = occupancy_grid
        self.grid_resolution = grid_resolution
        self.grid_origin = grid_origin
        # Pre‑compute values for fast grid conversion when an occupancy grid is provided.
        if occupancy_grid is not None:
            self._grid_shape = occupancy_grid.shape  # (rows, cols)
            self._inv_res = 1.0 / grid_resolution
            self._ox, self._oy = grid_origin
        else:
            self._grid_shape = (0, 0)
            self._inv_res = 0.0
            self._ox, self._oy = 0.0, 0.0

    # ------------------------------------------------------------------
    # Geometry helpers for collision checking
    # ------------------------------------------------------------------
    def _rectangle_occupied(self, cx: float, cy: float, heading: float, length: float, width: float) -> bool:
        """Check if any corner of a rectangle collides with an occupied cell.

        This version performs the corner calculations on‑the‑fly.  It will be
        accelerated by Numba's JIT compiler (see the ``_rectangle_occupied_numba``
        helper defined later in the file).
        """
        if self.occupancy_grid is None:
            return False
        return _rectangle_occupied_numba(
            cx,
            cy,
            heading,
            length,
            width,
            self.occupancy_grid,
            self._inv_res,
            self._ox,
            self._oy,
            self._grid_shape[0],  # rows
            self._grid_shape[1],  # cols
        )

    def plan(self, start: RobotState, goal: RobotState) -> List[RobotState]:
        """Compute a feasible path from *start* to *goal*.

        Returns
        -------
        List[RobotState]
            Ordered list of states from start to goal.  If no path is found,
            an empty list is returned.
        """
        # Priority queue stores entries of the form (f, g, tie_breaker, state, parent_key)
        # ``tie_breaker`` is a monotonically increasing integer to ensure that
        # entries are always comparable even when ``f`` and ``g`` are equal.
        open_queue: List[Tuple[float, float, int, RobotState, Tuple[int, int, int, int] | None]] = []
        start_key = _discretize(start)
        _push_counter = 0  # local counter for tie‑breaking
        heapq.heappush(open_queue, ( _heuristic(start, goal), 0.0, _push_counter, start, None ))
        _push_counter += 1

        # Maps discretised state key to (g_cost, parent_key, RobotState)
        closed: Dict[Tuple[int, int, int, int], Tuple[float, Tuple[int, int, int, int] | None, RobotState]] = {}
        closed[start_key] = (0.0, None, start)

        while open_queue:
            # The third element is the tie‑breaker counter; we ignore it for
            # algorithmic purposes.
            f, g, _tie, current, parent_key = heapq.heappop(open_queue)
            current_key = _discretize(current)

            # Goal test – we consider the goal reached when the robot position
            # is within a small tolerance.
            if math.hypot(current.x - goal.x, current.y - goal.y) < 0.2:
                # Reconstruct path.
                path: List[RobotState] = [current]
                back_key = current_key
                while True:
                    _, pred_key, pred_state = closed[back_key]
                    if pred_key is None:
                        break
                    path.append(pred_state)
                    back_key = pred_key
                path.reverse()
                return path

            # Expand neighbours using the provided control set.
            for v_left, v_right in self.control_set:
                neighbor = self.simulator.step(current, v_left, v_right, self.dt)
                # Collision check – skip this neighbour if it intersects an obstacle.
                if self.occupancy_grid is not None and self._collides(neighbor):
                    continue

                neighbor_key = _discretize(neighbor)
                tentative_g = g + self.dt  # cost proportional to time

                # If this node was never visited or we found a cheaper path.
                if neighbor_key not in closed or tentative_g < closed[neighbor_key][0]:
                    closed[neighbor_key] = (tentative_g, current_key, neighbor)
                    f_cost = tentative_g + _heuristic(neighbor, goal)
                    heapq.heappush(open_queue, (f_cost, tentative_g, _push_counter, neighbor, current_key))
                    _push_counter += 1
            # End of neighbour loop

        # No path found.
        return []

    # ------------------------------------------------------------------
    # Collision utilities (class methods, not nested inside ``plan``)
    # ------------------------------------------------------------------
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Fast conversion of a single world coordinate pair to grid indices.

        Uses the pre‑computed inverse resolution and origin stored during
        initialisation to avoid repeated division operations.
        """
        ix = int(math.floor((x - self._ox) * self._inv_res))
        iy = int(math.floor((y - self._oy) * self._inv_res))
        # Clip to grid bounds
        ix = max(0, min(ix, self._grid_shape[1] - 1))
        iy = max(0, min(iy, self._grid_shape[0] - 1))
        return ix, iy


    def _collides(self, state: RobotState) -> bool:
        """Return ``True`` if any part of the robot or trailer footprint
        intersects an occupied cell.

        This version leverages the fast ``_rectangle_occupied`` helper which
        combines corner generation and occupancy checking in a single tight
        loop, eliminating the need for a separate ``any_occupied`` function.
        """
        # Robot chassis
        if self._rectangle_occupied(
            state.x,
            state.y,
            state.theta_robot,
            self.simulator.params.robot_length,
            self.simulator.params.robot_width,
        ):
            return True

        # Trailer
        tx, ty, ttheta = self.simulator.trailer_pose(state)
        if self._rectangle_occupied(
            tx,
            ty,
            ttheta,
            self.simulator.params.trailer_length,
            self.simulator.params.trailer_width,
        ):
            return True

        return False

# ------------------------------------------------------------------
# Numba‑accelerated helper for rectangle occupancy checking
# ------------------------------------------------------------------
@njit
def _rectangle_occupied_numba(
    cx: float,
    cy: float,
    heading: float,
    length: float,
    width: float,
    occupancy_grid,
    inv_res: float,
    ox: float,
    oy: float,
    rows: int,
    cols: int,
):
    hl = length / 2.0
    hw = width / 2.0
    c = math.cos(heading)
    s = math.sin(heading)
    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        x = c * dx - s * dy + cx
        y = s * dx + c * dy + cy
        ix = int(math.floor((x - ox) * inv_res))
        iy = int(math.floor((y - oy) * inv_res))
        if ix < 0 or iy < 0 or ix >= cols or iy >= rows:
            continue
        if occupancy_grid[iy, ix]:
            return True
    return False

# ---------------------------------------------------------------------------
# Optional C++ accelerated implementation via pybind11
# ---------------------------------------------------------------------------
# The heavy‑weight planning calculations have been re‑implemented in C++ (see
# ``cpp/planner.cpp``) and exposed as the ``robot_trailer_cpp`` module. If the
# compiled extension is available we replace the pure‑Python ``HybridAStarPlanner``
# with the C++ version so the rest of the codebase can continue to import
# ``HybridAStarPlanner`` from this module without any changes.
try:
    from robot_trailer_cpp import HybridAStarPlanner as _CppPlanner
    # Preserve the original name for compatibility.
    HybridAStarPlanner = _CppPlanner  # type: ignore
    print("Using C++ planner implementation for better performance.")
except Exception:
    # If the C++ extension cannot be imported (e.g., not built), fall back to
    # the pure‑Python implementation defined above.
    print("Falling back to pure-Python planner implementation.  For better performance, build the C++ extension.")
    pass