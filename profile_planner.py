"""Profile the Hybrid A* planner.

This script reproduces the setup from ``run_planner.py`` but wraps the call to
``HybridAStarPlanner.plan`` in a profiler.  It writes three output files:

- ``planner_profile.prof`` – raw ``cProfile`` data
- ``planner_profile.txt`` – a human‑readable text report
- ``planner_profile.svg`` – a call‑graph / flame‑graph generated with
  ``gprof2dot`` and ``graphviz``

The script can be executed via the pixi environment, e.g.:

    pixi run python profile_planner.py --set default --start-x 0 --start-y 0 \
        --goal-x 5 --goal-y 5
"""

import argparse
import cProfile
import pstats
import subprocess
import sys
from pathlib import Path
import time

import numpy as np

from simulator import RobotParameters, Simulator, RobotState
from planner import HybridAStarPlanner

# ---------------------------------------------------------------------------
# Argument handling – mirror ``run_planner.py``
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Hybrid A* planner.")
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
# Control sets – identical to those in ``run_planner.py``
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
    # Occupancy grid (10 m × 10 m) with a few obstacles – same as ``run_planner``
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
    # Planner creation
    # ------------------------------------------------------------------
    planner = HybridAStarPlanner(
        simulator=sim,
        control_set=CONTROL_SETS[args.set],
        occupancy_grid=occupancy_grid,
        grid_resolution=grid_resolution,
        grid_origin=(0.0, 0.0),
    )

    # ------------------------------------------------------------------
    # Profile the planning step
    # ------------------------------------------------------------------
    #profiler = cProfile.Profile()
    #profiler.enable()
    starttime = time.time()
    path = planner.plan(start, goal)
    #endtime = time.time()
    #print(f"Planning took {endtime - starttime:.4f} seconds.")
    #profiler.disable()

    # ------------------------------------------------------------------
    # Save profiling data
    # ------------------------------------------------------------------
    #out_dir = Path.cwd()
    #prof_file = out_dir / "planner_profile.prof"
    #profiler.dump_stats(str(prof_file))

    # Human‑readable text report
    #txt_file = out_dir / "planner_profile.txt"
    #with txt_file.open("w") as f:
        #ps = pstats.Stats(profiler, stream=f)
        #ps.sort_stats(pstats.SortKey.CUMULATIVE)
        #ps.print_stats()

    # ------------------------------------------------------------------
    # Generate a call‑graph SVG using gprof2dot + dot (graphviz)
    # ------------------------------------------------------------------
    #svg_file = out_dir / "planner_profile_opt6.svg"
    #try:
        ## gprof2dot reads from stdin; we feed the .prof file via subprocess
        #gprof_cmd = ["gprof2dot", "-f", "pstats", str(prof_file)]
        #dot_cmd = ["dot", "-Tsvg", "-o", str(svg_file)]
        ## Pipe the output of gprof2dot into dot
        #gprof = subprocess.Popen(gprof_cmd, stdout=subprocess.PIPE)
        #dot = subprocess.Popen(dot_cmd, stdin=gprof.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #dot.communicate()
        #if dot.returncode != 0:
            #raise RuntimeError("dot returned non‑zero exit code")
    #except Exception as e:
        #print("Failed to generate SVG call‑graph:", e, file=sys.stderr)
        #print("You may need to install Graphviz (dot) and ensure it is on your PATH.")
#
    # ------------------------------------------------------------------
    # Summary for the user
    # ------------------------------------------------------------------
    print("Profiling complete.")
    #print(f"Raw profile data:   {prof_file}")
    #print(f"Text report:        {txt_file}")
    #print(f"Call‑graph SVG:      {svg_file}")
    #if not path:
        #print("No path was found for the given start/goal configuration.")


if __name__ == "__main__":
    main()
