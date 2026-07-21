/**
 * C++ implementation of the robot‑trailer kinematic simulator.
 *
 * The original pure‑Python implementation (see simulator.py) performed all
 * calculations in Python and used Numba for JIT acceleration.  To further
 * reduce overhead – especially when evaluating many motion primitives – the
 * core arithmetic is now implemented in C++ and exposed to Python via
 * pybind11.  The public API mirrors the Python version so existing code can be
 * switched by simply importing the new module.
 */

#include <cmath>
/** C++ implementation of the robot‑trailer kinematic simulator. */

#include "simulator.hpp"

/** Simple angle wrapping to the interval [-π, π]. */
static inline double wrap_angle(double angle) {
    double wrapped = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (wrapped < 0.0) wrapped += 2.0 * M_PI;
    return wrapped - M_PI;
}

// Implementations of the methods declared in simulator.hpp

Simulator::Simulator(const RobotParameters &params) : params_(params) {}

std::pair<double, double> Simulator::wheel_velocities_to_twist(double v_left, double v_right) const {
    double v = 0.5 * (v_left + v_right);
    double omega = (v_right - v_left) / params_.wheel_base;
    return {v, omega};
}

std::tuple<double, double, double, double>
Simulator::derivatives(const RobotState &state, double v_left, double v_right) const {
    auto [v, omega] = wheel_velocities_to_twist(v_left, v_right);
    double x_dot = v * std::cos(state.theta_robot);
    double y_dot = v * std::sin(state.theta_robot);
    double theta_dot = omega;
    double beta_dot = omega - (v / params_.drawbar_length) * std::sin(state.beta);
    return {x_dot, y_dot, theta_dot, beta_dot};
}

RobotState Simulator::step(const RobotState &state, double v_left, double v_right, double dt) const {
    auto [x_dot, y_dot, theta_dot, beta_dot] = derivatives(state, v_left, v_right);
    RobotState next;
    next.x = state.x + x_dot * dt;
    next.y = state.y + y_dot * dt;
    next.theta_robot = wrap_angle(state.theta_robot + theta_dot * dt);
    next.beta = wrap_angle(state.beta + beta_dot * dt);
    next.theta_trailer = next.theta_robot - next.beta;
    next.trailer_x = next.x - params_.drawbar_length * std::cos(next.theta_trailer);
    next.trailer_y = next.y - params_.drawbar_length * std::sin(next.theta_trailer);
    return next;
}

std::tuple<double, double, double> Simulator::trailer_pose(const RobotState &state) const {
    double theta_trailer = state.theta_robot - state.beta;
    double x_trailer = state.x - params_.drawbar_length * std::cos(theta_trailer);
    double y_trailer = state.y - params_.drawbar_length * std::sin(theta_trailer);
    return {x_trailer, y_trailer, theta_trailer};
}

std::vector<RobotState> Simulator::simulate(const RobotState &start,
                                            const std::vector<std::pair<double, double>> &controls,
                                            double dt) const {
    std::vector<RobotState> states;
    states.reserve(controls.size() + 1);
    states.push_back(start);
    RobotState cur = start;
    for (const auto &c : controls) {
        cur = step(cur, c.first, c.second, dt);
        states.push_back(cur);
    }
    return states;
}

const RobotParameters &Simulator::params() const { return params_; }
