"""Profile the C++ Hybrid A* planner.

This script mirrors ``profile_planner.py`` but focuses on the native C++
implementation that lives in the ``robot_trailer_cpp`` pybind11 module.  The
module must be built with profiling enabled (``-pg``) – the ``CMakeLists.txt``
has been updated to add the required flags.

The workflow is:

1. Build the C++ extensions with ``pixi run build_cpp`` (or the equivalent
   CMake command).  The ``-pg`` flag produces a ``gmon.out`` file after the
   program terminates.
2. Run this script.  It executes the planner (which now calls the C++
   implementation) and, once finished, invokes ``gprof`` on the generated
   ``gmon.out``.
3. Two reports are written to the current working directory:

   * ``cpp_planner_profile.txt`` – human‑readable flat profile produced by
     ``gprof``.
   * ``cpp_planner_profile.svg`` – a call‑graph generated via
     ``gprof2dot`` + Graphviz.

The script also prints the overall planning time measured with ``time.time``
so you can compare the pure‑Python and C++ versions.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from simulator import RobotParameters, Simulator, RobotState
from planner import HybridAStarPlanner  # This will resolve to the C++ version if built

# ---------------------------------------------------------------------------
# Argument handling – identical to ``profile_planner.py``
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the C++ Hybrid A* planner.")
    parser.add_argument(
        "--set",
        choices=["default", "slow_turn", "sharp_turn", "zigzag"],
        default="default",
        help="Control set used for motion primitives.",
    )
    parser.add_argument("--start-x", type=float, default=0.0, help="Start x position")
    parser.add_argument("--start-y", type=float, default=0.0, help="Start y position")
    parser.add_argument("--goal-x", type=float, default=5.0, help="Goal x position")
    parser.add_argument("--goal-y", type=float, default=5.0, help="Goal y position")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Control sets – copied from ``run_planner.py``
# ---------------------------------------------------------------------------
CONTROL_SETS = {
    "default": [(1.0, 1.0)] * 60 + [(0.5, 1.0)] * 50 + [(1.0, 1.0)] * 60,
    "slow_turn": [(0.5, 1.0)] * 80 + [(1.0, 0.5)] * 80,
    "sharp_turn": [(0.2, 1.0)] * 40 + [(1.0, 0.2)] * 40,
    "zigzag": [(1.0, 0.5)] * 30 + [(0.5, 1.0)] * 30 + [(1.0, 0.5)] * 30 + [(0.5, 1.0)] * 30,
}


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Simulator setup (same parameters as other scripts)
    # ------------------------------------------------------------------
    params = RobotParameters(wheel_base=0.6, drawbar_length=1.2)
    sim = Simulator(params)

    # ------------------------------------------------------------------
    # Define start/goal states
    # ------------------------------------------------------------------
    start = RobotState(x=args.start_x, y=args.start_y, theta_robot=0.0, beta=0.0)
    goal = RobotState(x=args.goal_x, y=args.goal_y, theta_robot=0.0, beta=0.0)

    # ------------------------------------------------------------------
    # Occupancy grid – identical to ``profile_planner.py``
    # ------------------------------------------------------------------
    grid_resolution = 0.1
    grid_size = int(10.0 / grid_resolution)
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=bool)

    def fill_rect(xmin, ymin, xmax, ymax):
        i_min = int(np.floor(xmin / grid_resolution))
        i_max = int(np.ceil(xmax / grid_resolution))
        j_min = int(np.floor(ymin / grid_resolution))
        j_max = int(np.ceil(ymax / grid_resolution))
        occupancy_grid[j_min:j_max, i_min:i_max] = True

    fill_rect(2.0, 2.0, 4.0, 4.0)
    fill_rect(6.0, 5.0, 8.0, 7.0)
    fill_rect(3.0, 7.0, 5.0, 9.0)

    # ------------------------------------------------------------------
    # Planner creation – will resolve to the C++ implementation if available
    # ------------------------------------------------------------------
    planner = HybridAStarPlanner(
        simulator=sim,
        control_set=CONTROL_SETS[args.set],
        occupancy_grid=occupancy_grid,
        grid_resolution=grid_resolution,
        grid_origin=(0.0, 0.0),
    )

    # ------------------------------------------------------------------
    # Run the planner while measuring wall‑clock time
    # ------------------------------------------------------------------
    start_time = time.time()
    path = planner.plan(start, goal)
    elapsed = time.time() - start_time
    print(f"Planning (C++) took {elapsed:.4f} seconds.")

    # ------------------------------------------------------------------
    # C++ profiling – ``gprof`` works on the shared object generated by
    # pybind11.  After the Python process exits a ``gmon.out`` file is written
    # to the current working directory.  We invoke ``gprof`` on that file and
    # generate both a flat text report and a call‑graph SVG.
    # ------------------------------------------------------------------
    # Determine the path to the compiled module (e.g. robot_trailer_cpp.cpython-311-x86_64-linux-gnu.so)
    cpp_module = importlib.import_module("robot_trailer_cpp")
    module_path = Path(cpp_module.__file__)  # type: ignore[arg-type]

    # Ensure any previous profiling data is removed so we analyse the fresh run.
    gmon_file = Path.cwd() / "gmon.out"
    if gmon_file.exists():
        gmon_file.unlink()

    # The Python interpreter will write ``gmon.out`` when it terminates.
    # We therefore need to spawn a separate process that re‑executes the same
    # call so that the file is available after the child exits.
    # To keep the script simple we re‑run the planner in a subprocess that
    # exits immediately after planning.
    subprocess.run([sys.executable, __file__, "--run-once", *sys.argv[1:]], check=False)

    # After the subprocess finishes, ``gmon.out`` should be present.
    if not gmon_file.is_file():
        print("gmon.out not found – ensure the C++ module was built with -pg and that the subprocess completed.")
        return

    out_dir = Path.cwd()
    txt_report = out_dir / "cpp_planner_profile.txt"
    svg_report = out_dir / "cpp_planner_profile.svg"

    # Flat profile (text)
    with txt_report.open("w") as f:
        subprocess.run(["gprof", str(module_path), str(gmon_file)], stdout=f, check=False)

    # Call‑graph via gprof2dot + dot
    try:
        gprof_cmd = ["gprof", str(module_path), str(gmon_file)]
        gprof2dot_cmd = ["gprof2dot", "-f", "gprof"]
        dot_cmd = ["dot", "-Tsvg", "-o", str(svg_report)]
        gprof = subprocess.Popen(gprof_cmd, stdout=subprocess.PIPE)
        g2d = subprocess.Popen(gprof2dot_cmd, stdin=gprof.stdout, stdout=subprocess.PIPE)
        dot = subprocess.Popen(dot_cmd, stdin=g2d.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = dot.communicate()
        if dot.returncode != 0:
            raise RuntimeError(err.decode())
    except Exception as e:  # pragma: no cover – optional visualisation
        print("Failed to generate SVG call‑graph:", e, file=sys.stderr)

    print("C++ profiling complete.")
    print(f"Flat profile:   {txt_report}")
    print(f"Call‑graph SVG: {svg_report}")
    if not path:
        print("No path was found for the given start/goal configuration.")


# ---------------------------------------------------------------------------
# Helper entry‑point used when the script re‑executes itself for profiling.
# ---------------------------------------------------------------------------
def _run_once() -> None:
    """Execute the planner a single time without any post‑processing.

    This function is invoked by the subprocess created in ``main``.  Its sole
    purpose is to let the Python interpreter generate ``gmon.out`` after the
    planner finishes.
    """
    args = parse_args()
    params = RobotParameters(wheel_base=0.6, drawbar_length=1.2)
    sim = Simulator(params)
    start = RobotState(x=args.start_x, y=args.start_y, theta_robot=0.0, beta=0.0)
    goal = RobotState(x=args.goal_x, y=args.goal_y, theta_robot=0.0, beta=0.0)
    grid_resolution = 0.1
    grid_size = int(10.0 / grid_resolution)
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=bool)
    def fill_rect(xmin, ymin, xmax, ymax):
        i_min = int(np.floor(xmin / grid_resolution))
        i_max = int(np.ceil(xmax / grid_resolution))
        j_min = int(np.floor(ymin / grid_resolution))
        j_max = int(np.ceil(ymax / grid_resolution))
        occupancy_grid[j_min:j_max, i_min:i_max] = True
    fill_rect(2.0, 2.0, 4.0, 4.0)
    fill_rect(6.0, 5.0, 8.0, 7.0)
    fill_rect(3.0, 7.0, 5.0, 9.0)
    planner = HybridAStarPlanner(
        simulator=sim,
        control_set=CONTROL_SETS[args.set],
        occupancy_grid=occupancy_grid,
        grid_resolution=grid_resolution,
        grid_origin=(0.0, 0.0),
    )
    planner.plan(start, goal)


if __name__ == "__main__":
    # ``--run-once`` is an internal flag; users should call the script without it.
    if "--run-once" in sys.argv:
        sys.argv.remove("--run-once")
        _run_once()
    else:
        main()
