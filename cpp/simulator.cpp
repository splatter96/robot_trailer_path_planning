#include <cmath>
#include <array>

#include "simulator.hpp"

/** Simple angle wrapping to the interval [-π, π]. */
static inline double wrap_angle(double angle) {
    if (angle > M_PI) {
        angle -= 2.0 * M_PI;
    } else if (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

static inline int fast_round(double val){
    return static_cast<int>(val + (val >= -1.0 ? 0.5 : -0.5));
    // return static_cast<int>(val + 0.5); // this only works if we have scrictly positive values, which we do for our discretisation indices
}

// Convert a continuous angle to a discretised index using the pre‑computed
// reciprocal of the angular resolution (inv_res_). The result is wrapped into
// the range [0, n) 
static inline int angle_to_index(double angle, double inv_res, int n) noexcept {
    int idx = static_cast<int>(std::floor(angle * inv_res + 0.5)); // round()
    // Ensure the index is positive before the modulo.
    if (idx < 0) idx += n * ((-idx) / n + 1);
    return idx % n;
}

Simulator::Simulator(const RobotParameters &params) : params_(params) {}

std::pair<double, double> Simulator::wheel_velocities_to_twist(double v_left, double v_right) const {
    double v = 0.5 * (v_left + v_right);
    double omega = (v_right - v_left) / params_.wheel_base;
    return {v, omega};
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
void Simulator::cache_discretization(RobotState &state) const {
    // Angular discretisation (used for derivative cache lookup).
    // state.theta_idx = angle_to_index(state.theta_robot, inv_res_, n_theta_);
    // state.beta_idx  = angle_to_index(state.beta,        inv_res_, n_beta_);


    // ---- angular discretisation (replace angle_to_index) -----------------
    // round the continuous angle to the nearest discrete step
    //int ti = static_cast<int>(std::round(state.theta_robot * inv_ang_res_));
    //int bi = static_cast<int>(std::round(state.beta        * inv_ang_res_));

    int ti = fast_round(state.theta_robot * inv_ang_res_);
    int bi = fast_round(state.beta        * inv_ang_res_);

    // wrap into [0, n_theta_) and [0, n_beta_)
    if (ti >= n_theta_)
        ti -= n_theta_;
    else if (ti < 0)
        ti += n_theta_;

    if (bi >= n_beta_)
        bi -= n_beta_;
    else if (bi < 0)
        bi += n_beta_;

    state.theta_idx = ti;
    state.beta_idx  = bi;

    // Position discretisation for fast collision checking and state indexing.
    double pos_res = pos_res_; // metres per grid cell
    state.ix   = fast_round(state.x * inv_pos_res_);
    state.iy   = fast_round(state.y * inv_pos_res_);
    state.ix_t = fast_round(state.trailer_x * inv_pos_res_);
    state.iy_t = fast_round(state.trailer_y * inv_pos_res_);
}

RobotState Simulator::step_cached(const RobotState &state, size_t control_idx, double dt) const {
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
    cache_discretization(next);
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
    for (size_t ctrl_idx = 0; ctrl_idx < controls.size(); ++ctrl_idx) {
        cur = step_cached(cur, ctrl_idx, dt);
        states.push_back(cur);
    }
    return states;
}

const RobotParameters &Simulator::params() const { return params_; }
