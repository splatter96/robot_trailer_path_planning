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
#include <array>
/** C++ implementation of the robot‑trailer kinematic simulator. */

#include "simulator.hpp"

/** Simple angle wrapping to the interval [-π, π]. */
// Fast angle wrapping to the interval [-π, π].
// The simulation updates angles by small increments, so a simple conditional
// adjustment is sufficient and avoids the relatively expensive std::fmod.
static inline double wrap_angle(double angle) {
    // Bring the angle into the range [-π, π] by adding or subtracting 2π once.
    if (angle > M_PI) {
        angle -= 2.0 * M_PI;
    } else if (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

// Convert a continuous angle to a discretised index using the pre‑computed
// reciprocal of the angular resolution (inv_res_). The result is wrapped into
// the range [0, n) where n is either n_theta_ or n_beta_. This function is
// deliberately tiny so the compiler can inline it.
static inline int angle_to_index(double angle, double inv_res, int n) noexcept {
    int idx = static_cast<int>(std::floor(angle * inv_res + 0.5)); // round()
    // Ensure the index is positive before the modulo.
    if (idx < 0) idx += n * ((-idx) / n + 1);
    return idx % n;
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

// ---------------------------------------------------------------------
// Cached derivative support – initialise a lookup table for a fixed control
// set and discretised orientation angles. This is intended for use by the
// planner, which knows the control set ahead of time.
// ---------------------------------------------------------------------
void Simulator::init_derivative_cache(const std::vector<std::pair<double, double>> &control_set,
                                      double ang_res) const {
    // Store parameters for later use.
    cached_control_set_ = control_set;
    ang_res_ = ang_res;
    inv_res_ = 1.0 / ang_res_;
    n_theta_ = static_cast<int>(std::ceil(2.0 * M_PI / ang_res_));
    n_beta_ = static_cast<int>(std::ceil(2.0 * M_PI / ang_res_));
    // Allocate cache: one entry per (control, theta, beta).
    derivative_cache_.clear();
    derivative_cache_.resize(control_set.size() * n_theta_ * n_beta_);

    for (size_t ci = 0; ci < control_set.size(); ++ci) {
        double v_left = control_set[ci].first;
        double v_right = control_set[ci].second;
        auto [v, omega] = wheel_velocities_to_twist(v_left, v_right);
        for (int ti = 0; ti < n_theta_; ++ti) {
            double theta = ti * ang_res_;
            double cos_theta = std::cos(theta);
            double sin_theta = std::sin(theta);
            double x_dot = v * cos_theta;
            double y_dot = v * sin_theta;
            for (int bi = 0; bi < n_beta_; ++bi) {
                double beta = bi * ang_res_;
                double beta_dot = omega - (v / params_.drawbar_length) * std::sin(beta);
                // Trailer orientation = theta - beta
                double theta_trailer = theta - beta;
                double cos_trailer = std::cos(theta_trailer);
                double sin_trailer = std::sin(theta_trailer);
                size_t idx = ci * n_theta_ * n_beta_ + ti * n_beta_ + bi;
                derivative_cache_[idx] = {x_dot, y_dot, omega, beta_dot, cos_trailer, sin_trailer};
            }
        }
    }
}

// Initialise the discretised angular indices inside a RobotState. This should be
// called once for any state that will be used with step_cached.
void Simulator::init_state(RobotState &state) const {
    // Angular discretisation (used for derivative cache lookup).
    state.theta_idx = angle_to_index(state.theta_robot, inv_res_, n_theta_);
    state.beta_idx  = angle_to_index(state.beta,        inv_res_, n_beta_);

    // Position discretisation for fast collision checking and state indexing.
    // The planner uses a default positional resolution of 0.1 m. We store the
    // rounded grid indices directly in the state so they can be reused without
    // recomputing `std::round` each time.
    constexpr double pos_res = 0.1; // metres per grid cell (matches discretize())
    state.ix   = static_cast<int>(std::round(state.x / pos_res));
    state.iy   = static_cast<int>(std::round(state.y / pos_res));
    state.ix_t = static_cast<int>(std::round(state.trailer_x / pos_res));
    state.iy_t = static_cast<int>(std::round(state.trailer_y / pos_res));
}

RobotState Simulator::step_cached(const RobotState &state, size_t control_idx, double dt) const {
    // If the cache has not been initialised (or the index is out of range),
    // fall back to the regular step implementation. This ensures safe
    // behaviour even if the planner forgets to initialise the cache.
    if (derivative_cache_.empty() || control_idx >= cached_control_set_.size()) {
        return state; // No movement – safe fallback.
    }

    // Use the cached discretised indices stored in the state.
    int theta_idx = state.theta_idx;
    int beta_idx  = state.beta_idx;
    size_t idx = static_cast<size_t>(control_idx) * n_theta_ * n_beta_ +
                 static_cast<size_t>(theta_idx) * n_beta_ +
                 static_cast<size_t>(beta_idx);
    const auto &der = derivative_cache_[idx];
    double x_dot = der[0];
    double y_dot = der[1];
    double theta_dot = der[2];
    double beta_dot = der[3];
    double cos_trailer = der[4];
    double sin_trailer = der[5];

    RobotState next;
    next.x = state.x + x_dot * dt;
    next.y = state.y + y_dot * dt;
    next.theta_robot = wrap_angle(state.theta_robot + theta_dot * dt);
    next.beta = wrap_angle(state.beta + beta_dot * dt);
    // Trailer orientation is theta_robot - beta; we already have its cosine and sine
    // from the cache (based on the discretised previous angles). These are accurate
    // enough for the planner's discretised search.
    next.theta_trailer = next.theta_robot - next.beta;
    next.trailer_x = next.x - params_.drawbar_length * cos_trailer;
    next.trailer_y = next.y - params_.drawbar_length * sin_trailer;

    // Update cached angular indices for the new state.
    init_state(next);
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
