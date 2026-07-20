from dataclasses import dataclass
import math
from numba import njit

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
@njit
def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

@njit
def _derivatives_numba(
    v: float,
    omega: float,
    theta_robot: float,
    beta: float,
    drawbar_length: float,
):
    """Compute the state derivatives for the robot‑trailer system.

    Returns a tuple ``(x_dot, y_dot, theta_dot, beta_dot)``.  This mirrors the
    logic previously inside ``Simulator.derivatives`` but is compiled with
    Numba for speed.
    """
    x_dot = v * math.cos(theta_robot)
    y_dot = v * math.sin(theta_robot)
    theta_dot = omega
    beta_dot = omega - (v / drawbar_length) * math.sin(beta)
    return (x_dot, y_dot, theta_dot, beta_dot)

@dataclass
class RobotParameters:
    """Physical parameters of the robot/trailer combination."""

    wheel_base: float  # Distance between left/right wheels [m]
    drawbar_length: float  # Hitch to trailer axle [m]
    # New geometric parameters used for visualization
    robot_length: float = 0.8  # Length of the robot chassis [m]
    robot_width: float = 0.6   # Width of the robot chassis [m]
    trailer_length: float = 1.2  # Length of the trailer [m]
    trailer_width: float = 0.8   # Width of the trailer [m]


@dataclass
class RobotState:
    """Robot state."""

    x: float
    y: float
    theta_robot: float  # Heading of robot [rad]
    beta: float  # Articulation angle = robot - trailer [rad]


class Simulator:
    """
    Kinematic simulator for a skid-steered robot pulling a single trailer.

    Assumptions
    -----------
    - Differential drive approximation
    - Hitch located exactly at robot rotation center
    - No wheel slip
    - Single axle trailer
    """

    def __init__(self, params: RobotParameters):
        self.params = params

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def wheel_velocities_to_twist(self, v_left: float, v_right: float):
        """
        Convert wheel velocities into forward and angular velocity.
        """

        v = 0.5 * (v_left + v_right)

        omega = (v_right - v_left) / self.params.wheel_base

        return v, omega

    def derivatives(self, state: RobotState, v_left: float, v_right: float):
        """Compute state derivatives using a Numba‑accelerated helper.

        The original implementation performed a method call to
        ``wheel_velocities_to_twist`` and several ``math`` function calls per
        evaluation.  To reduce overhead we inline the calculations and delegate
        the core arithmetic to a ``@njit`` compiled function which operates on
        plain ``float`` values, avoiding attribute look‑ups and Python call
        overhead.
        """
        # Inline wheel velocity conversion
        v = 0.5 * (v_left + v_right)
        omega = (v_right - v_left) / self.params.wheel_base

        return _derivatives_numba(
            v,
            omega,
            state.theta_robot,
            state.beta,
            self.params.drawbar_length,
        )

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def step(self, state: RobotState, v_left: float, v_right: float, dt: float):
        """
        Propagate one Euler integration step.
        """

        x_dot, y_dot, theta_dot, beta_dot = self.derivatives(
            state,
            v_left,
            v_right,
        )

        return RobotState(
            x=state.x + x_dot * dt,
            y=state.y + y_dot * dt,
            theta_robot=wrap_angle(state.theta_robot + theta_dot * dt),
            beta=wrap_angle(state.beta + beta_dot * dt),
        )

    # ------------------------------------------------------------------
    # Derived trailer pose
    # ------------------------------------------------------------------
    def trailer_pose(self, state: RobotState):
        """
        Compute trailer axle pose.
        """

        theta_trailer = state.theta_robot - state.beta

        x_trailer = state.x - self.params.drawbar_length * math.cos(theta_trailer)

        y_trailer = state.y - self.params.drawbar_length * math.sin(theta_trailer)

        return (
            x_trailer,
            y_trailer,
            theta_trailer,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def simulate(self, state: RobotState, controls, dt):
        """
        Simulate multiple control inputs.

        controls:
            iterable of (v_left, v_right)
        """

        states = [state]

        current = state

        for v_left, v_right in controls:
            current = self.step(
                current,
                v_left,
                v_right,
                dt,
            )
            states.append(current)

        return states

# ---------------------------------------------------------------------------
# Optional C++ accelerated implementation via pybind11
# ---------------------------------------------------------------------------
# The heavy‑weight calculations (wheel velocity conversion, derivative
# computation, and integration) have been re‑implemented in C++ (see
# ``cpp/simulator.cpp``) and exposed as the ``robot_trailer_sim_cpp`` module.
# If the compiled extension is available we replace the pure‑Python classes
# with the C++ versions so the rest of the codebase can continue to import
# ``Simulator``, ``RobotParameters`` and ``RobotState`` from this module without
# any changes.
try:
    from robot_trailer_sim_cpp import Simulator as _CppSimulator
    from robot_trailer_sim_cpp import RobotParameters as _CppRobotParameters
    from robot_trailer_sim_cpp import RobotState as _CppRobotState

    # Preserve the original names for compatibility.
    Simulator = _CppSimulator  # type: ignore
    RobotParameters = _CppRobotParameters  # type: ignore
    RobotState = _CppRobotState  # type: ignore
    print("Using C++ simulator implementation for better performance.")
except Exception:
    # If the C++ extension cannot be imported (e.g., not built), fall back to
    # the pure‑Python implementation defined above.
    print("Falling back to pure-Python simulator implementation.  For better performance, build the C++ extension.")
    pass