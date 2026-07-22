#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>

#include "simulator.hpp"
#include "planner.hpp"

using Clock = std::chrono::steady_clock;

// Simple command-line parsing helper.
struct Args {
    std::string control_set = "default";
    double start_x = 0.0;
    double start_y = 0.0;
    double goal_x = 5.0;
    double goal_y = 5.0;
};

Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--set" && i + 1 < argc) {
            a.control_set = argv[++i];
        } else if (s == "--start-x" && i + 1 < argc) {
            a.start_x = std::stod(argv[++i]);
        } else if (s == "--start-y" && i + 1 < argc) {
            a.start_y = std::stod(argv[++i]);
        } else if (s == "--goal-x" && i + 1 < argc) {
            a.goal_x = std::stod(argv[++i]);
        } else if (s == "--goal-y" && i + 1 < argc) {
            a.goal_y = std::stod(argv[++i]);
        }
    }
    return a;
}

int main(int argc, char **argv) {
    auto args = parse_args(argc, argv);

    // Define control sets (mirror run_planner.py)
    std::map<std::string, std::vector<std::pair<double,double>>> CONTROL_SETS;
    CONTROL_SETS["default"] = {{1.0,1.0},{0.5,1.0},{1.0,0.5}};
    CONTROL_SETS["slow_turn"] = std::vector<std::pair<double,double>>(160, {0.5,1.0});
    // slow_turn in python was 80 + 80 with two different pairs; approximate
    CONTROL_SETS["slow_turn"].assign(80, {0.5,1.0});
    for (int i=0;i<80;i++) CONTROL_SETS["slow_turn"].push_back({1.0,0.5});
    CONTROL_SETS["sharp_turn"] = std::vector<std::pair<double,double>>(80, {0.2,1.0});
    CONTROL_SETS["zigzag"] = {};
    for (int i=0;i<30;i++) { CONTROL_SETS["zigzag"].push_back({1.0,0.5}); }
    for (int i=0;i<30;i++) { CONTROL_SETS["zigzag"].push_back({0.5,1.0}); }
    for (int i=0;i<30;i++) { CONTROL_SETS["zigzag"].push_back({1.0,0.5}); }
    for (int i=0;i<30;i++) { CONTROL_SETS["zigzag"].push_back({0.5,1.0}); }
    // dense: linspace(-1,1,3) x linspace(-1,1,3)
    for (double vl : {-1.0, 0.0, 1.0}) {
        for (double vr : {-1.0, 0.0, 1.0}) {
            CONTROL_SETS["dense"].push_back({vl, vr});
        }
    }

    if (CONTROL_SETS.find(args.control_set) == CONTROL_SETS.end()) {
        std::cerr << "Unknown control set: " << args.control_set << std::endl;
        return 1;
    }

    // Simulator parameters (match Python)
    RobotParameters params;
    params.wheel_base = 0.6;
    params.drawbar_length = 1.2;
    Simulator sim(params);

    RobotState start{args.start_x, args.start_y, 0.0, 0.0};
    RobotState goal{args.goal_x, args.goal_y, 0.0, 0.0};

    // Occupancy grid: 10m x 10m
    double grid_resolution = 0.5;
    int grid_size = static_cast<int>(std::floor(10.0 / grid_resolution));
    int rows = grid_size;
    int cols = grid_size;
    std::vector<uint8_t> occupancy(rows * cols, 0);

    double angular_resolution =  M_PI / 36.0;

    auto fill_rect = [&](double xmin, double ymin, double xmax, double ymax) {
        int i_min = static_cast<int>(std::floor(xmin / grid_resolution));
        int i_max = static_cast<int>(std::ceil(xmax / grid_resolution));
        int j_min = static_cast<int>(std::floor(ymin / grid_resolution));
        int j_max = static_cast<int>(std::ceil(ymax / grid_resolution));
        i_min = std::max(0, i_min);
        j_min = std::max(0, j_min);
        i_max = std::min(cols, i_max);
        j_max = std::min(rows, j_max);
        for (int j = j_min; j < j_max; ++j) {
            for (int i = i_min; i < i_max; ++i) {
                occupancy[j * cols + i] = 1;
            }
        }
    };

    fill_rect(2.0, 2.0, 4.0, 4.0);
    fill_rect(6.0, 5.0, 8.0, 7.0);
    fill_rect(3.0, 7.0, 5.0, 9.0);

    std::cout << "Using control set: " << args.control_set << std::endl;

    auto start_init = Clock::now();
    HybridAStarPlanner planner(sim, CONTROL_SETS[args.control_set], occupancy, rows, cols, grid_resolution, angular_resolution, {0.0,0.0}, 0.1);
    auto end_init = Clock::now();
    std::chrono::duration<double> init_dur = end_init - start_init;
    std::cout << "Planner initialisation took " << init_dur.count() << " seconds." << std::endl;

    auto t0 = Clock::now();
    auto path = planner.plan(start, goal);
    auto t1 = Clock::now();
    std::chrono::duration<double> plan_dur = t1 - t0;
    std::cout << "Planning took " << plan_dur.count() << " seconds." << std::endl;

    if (path.empty()) {
        std::cout << "No feasible path found." << std::endl;
        return 0;
    }

    // Write path to CSV
    std::ofstream ofs("path.csv");
    ofs << "x,y,theta_robot,beta\n";
    for (const auto &s : path) {
        ofs << s.x << "," << s.y << "," << s.theta_robot << "," << s.beta << "\n";
    }
    ofs.close();
    std::cout << "Wrote path with " << path.size() << " states to path.csv" << std::endl;

    return 0;
}
