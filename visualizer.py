import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from dataclasses import dataclass


@dataclass
class VisualizationParameters:
    # Drawing options (dimensions are now stored in RobotParameters)
    draw_heading: bool = True
    heading_length: float = 0.5


class Visualizer:
    def __init__(
        self,
        simulator,
        vis_params,
        occupancy_grid=None,
        grid_resolution: float = 0.1,
        grid_origin: tuple[float, float] = (0.0, 0.0),
    ):
        """Create a visualiser.

        Parameters
        ----------
        simulator: Simulator
            The kinematic simulator providing ``trailer_pose``.
        vis_params: VisualizationParameters
            Visual styling parameters.
        occupancy_grid: np.ndarray | None, optional
            Boolean grid where ``True`` indicates an obstacle. If ``None`` the
            grid is not drawn.
        grid_resolution: float, optional
            Size of a grid cell in world metres (default 0.1 m).
        grid_origin: tuple(float, float), optional
            World coordinates of the lower‑left corner of the grid.
        """
        self.simulator = simulator
        self.params = vis_params
        self.occupancy_grid = occupancy_grid
        self.grid_resolution = grid_resolution
        self.grid_origin = grid_origin

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------

    @staticmethod
    def _rectangle(center_x, center_y, heading, length, width):
        """
        Returns the four corners of a rotated rectangle.
        """

        hl = length / 2.0
        hw = width / 2.0

        corners = [
            (+hl, +hw),
            (+hl, -hw),
            (-hl, -hw),
            (-hl, +hw),
        ]

        c = math.cos(heading)
        s = math.sin(heading)

        result = []

        for x, y in corners:
            xr = c * x - s * y + center_x
            yr = s * x + c * y + center_y

            result.append((xr, yr))

        return result

    # ------------------------------------------------------------
    # Draw one state
    # ------------------------------------------------------------

    def draw(self, ax, state, history=None, explored=None):
        """
        Draw robot and trailer.
        """

        ax.clear()

        # --------------------------------------------------------
        # Occupancy grid (if provided)
        # --------------------------------------------------------
        if self.occupancy_grid is not None:
            # ``imshow`` expects the first dimension to correspond to the y‑axis.
            # The grid is stored as [row, col] where row indexes y.
            height = self.occupancy_grid.shape[0] * self.grid_resolution
            width = self.occupancy_grid.shape[1] * self.grid_resolution
            ox, oy = self.grid_origin
            ax.imshow(
                self.occupancy_grid,
                origin="lower",
                extent=(ox, ox + width, oy, oy + height),
                cmap="Greys",
                alpha=0.5,
            )

        # --------------------------------------------------------
        # Robot
        # --------------------------------------------------------

        # Robot dimensions are now stored in ``simulator.params``.
        robot_poly = self._rectangle(
            state.x,
            state.y,
            state.theta_robot,
            self.simulator.params.robot_length,
            self.simulator.params.robot_width,
        )

        ax.add_patch(
            Polygon(
                robot_poly,
                closed=True,
                fill=False,
                linewidth=2,
            )
        )

        # --------------------------------------------------------
        # Trailer
        # --------------------------------------------------------

        trailer_x, trailer_y, trailer_theta = self.simulator.trailer_pose(state)

        # Trailer dimensions are also stored in ``simulator.params``.
        trailer_poly = self._rectangle(
            trailer_x,
            trailer_y,
            trailer_theta,
            self.simulator.params.trailer_length,
            self.simulator.params.trailer_width,
        )

        ax.add_patch(
            Polygon(
                trailer_poly,
                closed=True,
                fill=False,
                linewidth=2,
            )
        )

        # --------------------------------------------------------
        # Draw drawbar
        # --------------------------------------------------------

        ax.plot(
            [state.x, trailer_x],
            [state.y, trailer_y],
            linewidth=2,
        )

        # --------------------------------------------------------
        # Draw hitch
        # --------------------------------------------------------

        ax.plot(
            state.x,
            state.y,
            marker="o",
            markersize=6,
        )

        # --------------------------------------------------------
        # Heading arrow
        # --------------------------------------------------------

        if self.params.draw_heading:
            hx = state.x + self.params.heading_length * math.cos(state.theta_robot)

            hy = state.y + self.params.heading_length * math.sin(state.theta_robot)

            ax.plot(
                [state.x, hx],
                [state.y, hy],
                linewidth=2,
            )

        # --------------------------------------------------------
        # Path history
        # --------------------------------------------------------

        if history is not None:
            ax.plot(
                [s.x for s in history],
                [s.y for s in history],
                linewidth=1,
            )

        # Visualise all nodes that were expanded during planning.  These are
        # typically a large set, so we plot them as semi‑transparent small
        # markers to avoid cluttering the view.
        if explored is not None:
            ax.scatter(
                [s.x for s in explored],
                [s.y for s in explored],
                c="red",
                s=4,
                alpha=0.4,
                label="explored",
            )

        # --------------------------------------------------------
        # Plot settings
        # --------------------------------------------------------

        ax.set_aspect("equal")

        ax.grid(True)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        ax.autoscale_view()
