#pragma once

#include "simulator.hpp"
#include <vector>
#include <utility>
#include <string>
#include <memory>
// #include "footprint_lookup.hpp"
#include "profiler.hpp"

// Hybrid A* planner (declaration). The implementation is in planner.cpp and
// is independent of pybind11. Bindings are provided in bindings.cpp.
class HybridAStarPlanner {
public:
    // The planner needs to initialise the simulator's derivative cache.
    // The simulator is stored as a const reference; cache members are mutable.
    HybridAStarPlanner(const Simulator &sim,
                       const std::vector<std::pair<double, double>> &control_set,
                       const std::vector<uint8_t> &occupancy_grid = std::vector<uint8_t>(),
                       int rows = 0,
                       int cols = 0,
                       double grid_resolution = 0.1,
                       std::pair<double, double> grid_origin = {0.0, 0.0},
                       double dt = 0.1);

    void reset_explored();
    std::vector<RobotState> get_explored() const;

    // Return a flattened distance map (row-major) of size rows*cols. If no
    // occupancy grid was provided an empty vector is returned.
    std::vector<double> reverse_distance_map(const RobotState &goal) const;

    std::vector<RobotState> plan(const RobotState &start, const RobotState &goal);
    std::string profile_summary() const;

    int rows() const;
    int cols() const;

private:
    const Simulator &simulator_;
    std::vector<std::pair<double, double>> control_set_;
    double dt_;
    std::vector<RobotState> explored_;
    std::vector<uint8_t> occupancy_grid_;
    double grid_resolution_;
    double ox_, oy_;
    double inv_res_ = 0.0;
    int rows_ = 0, cols_ = 0;

    // Fast lookup tables for robot and trailer footprints (kept for possible future extensions).
    // std::unique_ptr<FootprintLookup> robot_lookup_;
    // std::unique_ptr<FootprintLookup> trailer_lookup_;

    // Distance map (Euclidean distance to nearest obstacle) and clearance threshold.
    std::vector<double> distance_grid_; // flattened row-major
    double clearance_threshold_ = 0.0;

    // Lightweight profiler for timing hot regions.
    Profiler profiler_;

    bool collides(const RobotState &state) const;
};
