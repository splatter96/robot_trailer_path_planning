"""Run the Hybrid A* planner and visualise the resulting path.

This script creates a start and goal ``RobotState`` for the skid‑steer robot
with a trailer, runs the ``HybridAStarPlanner`` defined in ``planner.py`` and
visualises the resulting trajectory using the same ``Visualizer`` class that
is used by ``run_visualizer.py``.
"""

import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.animation as animation

from simulator import RobotParameters, Simulator, RobotState
from visualizer import VisualizationParameters, Visualizer
from planner import HybridAStarPlanner

# ---------------------------------------------------------------------------
# Define a few simple control sets (must match those used in the visualiser)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Control sets (motion primitives) used by the planner.
# ---------------------------------------------------------------------------
# The existing sets are simple sequences useful for demos.  The new "dense"
# set provides a broader sampling of left/right wheel velocity pairs, making the
# planner applicable to a wider variety of start/goal configurations and obstacle
# layouts.  It includes forward, reverse and turning motions.
CONTROL_SETS = {
    # "default": [(1.0, 1.0)] * 60 + [(0.5, 1.0)] * 50 + [(1.0, 1.0)] * 60,
    "default": [(1.0, 1.0)]  + [(0.5, 1.0)]  + [(1.0, 0.5)] ,
    "slow_turn": [(0.5, 1.0)] * 80 + [(1.0, 0.5)] * 80,
    "sharp_turn": [(0.2, 1.0)] * 40 + [(1.0, 0.2)] * 40,
    "zigzag": [(1.0, 0.5)] * 30 + [(0.5, 1.0)] * 30 + [(1.0, 0.5)] * 30 + [(0.5, 1.0)] * 30,
    # "dense" provides a grid of velocity pairs covering forward, reverse and
    # turning motions.  The step size (0.5) yields a manageable number of
    # primitives while still offering good coverage.
    "dense": [
        (vl, vr)
        for vl in np.linspace(-1.0, 1.0, 3)  # -1.0, -0.75, ..., 1.0
        for vr in np.linspace(-1.0, 1.0, 3)
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid A* planner demo.")
    parser.add_argument(
        "--set",
        choices=CONTROL_SETS.keys(),
        default="default",
        help="Control set used by the planner for motion primitives.",
    )
    # Simple start/goal specification – users can provide x, y coordinates.
    parser.add_argument("--start-x", type=float, default=0.0, help="Start x position")
    parser.add_argument("--start-y", type=float, default=0.0, help="Start y position")
    parser.add_argument("--goal-x", type=float, default=5.0, help="Goal x position")
    parser.add_argument("--goal-y", type=float, default=5.0, help="Goal y position")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Simulator and visualiser setup (same parameters as run_visualizer)
    # ------------------------------------------------------------------
    params = RobotParameters(wheel_base=0.6, drawbar_length=1.2)
    sim = Simulator(params)
    vis_params = VisualizationParameters()
    # Pass the occupancy grid to the visualizer so it can be drawn as a background.
    # ------------------------------------------------------------------
    # Define start and goal states (heading and articulation start at 0)
    # ------------------------------------------------------------------
    start = RobotState(x=args.start_x, y=args.start_y, theta_robot=0.0, beta=0.0)
    goal = RobotState(x=args.goal_x, y=args.goal_y, theta_robot=0.0, beta=0.0)

    # ------------------------------------------------------------------
    # Create a simple 10 m × 10 m occupancy grid with a few rectangular obstacles.
    # ------------------------------------------------------------------
    grid_resolution = 0.1  # metres per cell
    grid_size = int(10.0 / grid_resolution)  # 100 cells per side
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=bool)

    # Helper to fill a rectangular region (xmin, ymin, xmax, ymax) in world coords.
    def fill_rect(xmin, ymin, xmax, ymax):
        i_min = int(np.floor(xmin / grid_resolution))
        i_max = int(np.ceil(xmax / grid_resolution))
        j_min = int(np.floor(ymin / grid_resolution))
        j_max = int(np.ceil(ymax / grid_resolution))
        occupancy_grid[j_min:j_max, i_min:i_max] = True

    # Define some obstacles (all units in metres).
    fill_rect(2.0, 2.0, 4.0, 4.0)   # square obstacle
    fill_rect(6.0, 5.0, 8.0, 7.0)   # another square
    fill_rect(3.0, 7.0, 5.0, 9.0)   # vertical rectangle

    vis = Visualizer(
        sim,
        vis_params,
        occupancy_grid=occupancy_grid,
        grid_resolution=grid_resolution,
        grid_origin=(0.0, 0.0),
    )

    # ------------------------------------------------------------------
    # Planner initialisation (with occupancy grid) and distance‑map visualisation
    # ------------------------------------------------------------------
    print(f"Using control set: {CONTROL_SETS[args.set]}")
    starttime = time.time()
    planner = HybridAStarPlanner(
        simulator=sim,
        control_set=CONTROL_SETS[args.set],
        occupancy_grid=occupancy_grid,
        grid_resolution=grid_resolution,
        grid_origin=(0.0, 0.0),
        angular_resolution=np.pi / 36.0,  # 5 degrees
    )
    endtime = time.time()
    print(f"Planner initialisation took {endtime - starttime:.4f} seconds.")

    # Compute the reverse distance‑to‑goal map (heuristic field) and display it.
    # This is shown before the planner expands any nodes.
    # distance_map = planner.reverse_distance_map(goal)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # if distance_map.size > 0:
    #     rows, cols = distance_map.shape
    #     extent = (
    #         0.0,
    #         cols * grid_resolution,
    #         0.0,
    #         rows * grid_resolution,
    #     )
    #     ax.imshow(
    #         distance_map,
    #         origin="lower",
    #         extent=extent,
    #         cmap="viridis",
    #         alpha=0.6,
    #     )
    #     # Add a colour bar for reference.
    #     cbar = fig.colorbar(
    #         plt.cm.ScalarMappable(cmap="viridis"),
    #         ax=ax,
    #         label="Distance to goal (m)",
    #     )
    #     cbar.set_alpha(1.0)
    #     #cbar.draw_all()
    # plt.show()

    # Now run the actual planning (timed).
    starttime = time.time()
    path = planner.plan(start, goal)
    # Retrieve all nodes that were expanded during planning for visualisation.
    explored_nodes = planner.get_explored()
    print(f"Number of explored nodes: {len(explored_nodes)}")
    endtime = time.time()
    print(f"Planning took {endtime - starttime:.4f} seconds.")

    if not path:
        print("No feasible path found.")
        return

    # print(planner.profile_summary())

    # ------------------------------------------------------------------
    # Visualise the resulting path with animation (same style as run_visualizer)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # Simulation time step – used only for animation timing.
    dt = 0.1

    def init():
        """Draw the initial state before the animation starts."""
        # At frame 0 we have only the start state.
        # Visualise the start state, the path history, and the explored nodes.
        vis.draw(ax, path[0], [path[0]], explored=explored_nodes)
        # vis.draw(ax, path[0], [path[0]])
        return []

    def update(frame):
        """Update function for each animation frame.

        ``frame`` indexes into the ``path`` list produced by the planner.
        """
        # Current state and the history up to this frame.
        state = path[frame]
        history = path[: frame + 1]
        vis.draw(ax, state, history, explored=explored_nodes)
        # vis.draw(ax, state, history)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(path),
        init_func=init,
        interval=dt * 1000,
        repeat=False,
    )

    ax.set_title("Hybrid A* planned path (animated)")
    plt.show()


if __name__ == "__main__":
    main()
