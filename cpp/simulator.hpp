// Header declaring shared data structures and the Simulator class.
// This header is used by both simulator.cpp (which provides the definitions)
// and planner.cpp (which needs the declarations).

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// Data structures – identical to those defined in simulator.cpp.
// ---------------------------------------------------------------------------
struct RobotParameters {
    double wheel_base;          // distance between wheels [m]
    double drawbar_length;      // hitch to trailer axle [m]
    double robot_length = 0.8;  // robot chassis length [m]
    double robot_width = 0.6;   // robot chassis width  [m]
    double trailer_length = 1.2; // trailer length [m]
    double trailer_width = 0.8;  // trailer width  [m]
};

struct RobotState {
    double x;            // world x position (m)
    double y;            // world y position (m)
    double theta_robot;  // robot heading (rad)
    double beta;         // articulation angle (rad)
    // Trailer pose – cached here for quick access after each step.
    double trailer_x = 0.0;      // trailer world x (m)
    double trailer_y = 0.0;      // trailer world y (m)
    double theta_trailer = 0.0;  // trailer heading (rad)
};

// ---------------------------------------------------------------------------
// Simulator class declaration – definitions are in simulator.cpp.
// ---------------------------------------------------------------------------
class Simulator {
public:
    explicit Simulator(const RobotParameters &params);
    std::pair<double, double> wheel_velocities_to_twist(double v_left, double v_right) const;
    std::tuple<double, double, double, double> derivatives(const RobotState &state, double v_left, double v_right) const;
    RobotState step(const RobotState &state, double v_left, double v_right, double dt) const;
    std::tuple<double, double, double> trailer_pose(const RobotState &state) const;
    const RobotParameters &params() const;

private:
    RobotParameters params_;
};
