import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import RobotParameters, Simulator, RobotState
from visualizer import VisualizationParameters, Visualizer

params = RobotParameters(
    wheel_base=0.6,
    drawbar_length=1.2,
)

vis_params = VisualizationParameters()

sim = Simulator(params)
vis = Visualizer(sim, vis_params)

state = RobotState(
    x=0,
    y=0,
    theta_robot=0,
    beta=0,
)

# ------------------------------------------------------------------
# Define multiple feasible control command sets
# ------------------------------------------------------------------

control_sets = {
    "default": [(1.0, 1.0)] * 60 + [(0.5, 1.0)] * 50 + [(1.0, 1.0)] * 60,
    "slow_turn": [(0.5, 1.0)] * 80 + [(1.0, 0.5)] * 80,
    "sharp_turn": [(0.2, 1.0)] * 40 + [(1.0, 0.2)] * 40,
    "zigzag": [(1.0, 0.5)] * 30 + [(0.5, 1.0)] * 30 + [(1.0, 0.5)] * 30 + [(0.5, 1.0)] * 30,
}

parser = argparse.ArgumentParser(description="Run robot‑trailer simulation with selectable control set.")
parser.add_argument(
    "--set",
    choices=control_sets.keys(),
    default="default",
    help="Select which predefined control command set to use.",
)
args = parser.parse_args()

# Initialize history with the starting state.
history = [state]

# Choose the control sequence based on the user argument.
controls = control_sets[args.set]

fig, ax = plt.subplots(figsize=(8, 8))

# ------------------------------------------------------------------
# Animation
# ------------------------------------------------------------------

# Simulation time step (seconds)
dt = 0.1

def init():
    """Initial drawing before the animation starts."""
    vis.draw(ax, state, history)
    return []

def update(frame):
    """Update function for each animation frame.

    ``frame`` is the index into the ``controls`` list.
    """
    global state
    # Apply the control for this frame.
    v_left, v_right = controls[frame]
    state = sim.step(state, v_left, v_right, dt)
    history.append(state)
    # Redraw the visualizer with the new state and updated history.
    vis.draw(ax, state, history)
    return []

# Create the animation. Interval is dt (seconds) converted to milliseconds.
anim = animation.FuncAnimation(
    fig,
    update,
    frames=len(controls),
    init_func=init,
    interval=dt * 1000,
    repeat=False,
)

plt.show()
