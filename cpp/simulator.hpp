#pragma once

#include <tuple>
#include <vector>
#include <array>
#include <cmath>

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
    // Discretised angular indices – cached for fast lookup in the derivative table.
    // They correspond to round(theta / ang_res) and round(beta / ang_res).
    int theta_idx = 0;
    int beta_idx = 0;
    // Trailer pose – cached here for quick access after each step.
    double trailer_x = 0.0;      // trailer world x (m)
    double trailer_y = 0.0;      // trailer world y (m)
    double theta_trailer = 0.0;  // trailer heading (rad)
    // Discretised grid indices for fast collision checking. -1 indicates not yet computed.
    // Marked mutable so they can be updated even when the RobotState is passed as const.
    mutable int ix = -1;   // robot centre grid x index
    mutable int iy = -1;   // robot centre grid y index
    mutable int ix_t = -1; // trailer centre grid x index
    mutable int iy_t = -1; // trailer centre grid y index
};

class Simulator {
public:
    explicit Simulator(const RobotParameters &params);
    std::pair<double, double> wheel_velocities_to_twist(double v_left, double v_right) const;
    std::tuple<double, double, double> trailer_pose(const RobotState &state) const;
    const RobotParameters &params() const;
    std::vector<RobotState> simulate(const RobotState &start, const std::vector<std::pair<double, double>> &controls, double dt) const;

    // ---------------------------------------------------------------------
    // Performance helpers – pre‑compute derivatives for a known control set.
    // The planner can initialise the cache once and then use `step_cached`
    // for fast state propagation during the search.
    // ---------------------------------------------------------------------
    void init_derivative_cache(const std::vector<std::pair<double, double>> &control_set,
                               double ang_res = M_PI / 36.0) const;
    // Initialise the discretised angular indices inside a RobotState.
    void cache_discretization(RobotState &state) const;
    RobotState step_cached(const RobotState &state, size_t control_idx, double dt) const;

    // Set the positional resolution used for discretising world coordinates.
    void set_grid_resolution(double res) const {
         pos_res_ = res; 
         inv_pos_res_ = (res != 0.0) ? 1.0 / res : 0.0;
        }

    // Set the angular resolution (radians per discretisation step) used for
    // indexing orientation angles. The inverse resolution is cached for fast
    // conversion to indices.
    void set_angular_resolution(double res) const {
        ang_res_ = res;
        // Update the reciprocal used by angle_to_index.
        inv_ang_res_ = (res != 0.0) ? 1.0 / res : 0.0;
    }

private:
    RobotParameters params_;
    // Cached derivative table: indexed by [control][theta_index][beta_index]
    // each entry stores {x_dot, y_dot, theta_dot, beta_dot, cos_trailer, sin_trailer}.
    mutable std::vector<std::array<double, 6>> derivative_cache_;
    mutable int n_theta_ = 0;
    mutable int n_beta_ = 0;
    mutable double ang_res_ = M_PI / 36.0;
    mutable double inv_ang_res_ = 0.0; // reciprocal of ang_res_ for fast index conversion
    mutable double pos_res_ = 0.1;
    mutable double inv_pos_res_ = 0.0;
    mutable std::vector<std::pair<double, double>> cached_control_set_;
};
