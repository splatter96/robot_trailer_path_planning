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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;

/**
 * Simple angle wrapping to the interval [-π, π].
 */
static inline double wrap_angle(double angle) {
    // fmod can return negative values; adding 2π ensures positivity before
    // final subtraction.
    double wrapped = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (wrapped < 0.0) wrapped += 2.0 * M_PI;
    return wrapped - M_PI;
}

/**
 * Physical parameters of the robot and trailer.
 */
struct RobotParameters {
    double wheel_base;          // distance between wheels [m]
    double drawbar_length;      // hitch to trailer axle [m]
    double robot_length = 0.8;  // robot chassis length [m]
    double robot_width = 0.6;   // robot chassis width  [m]
    double trailer_length = 1.2; // trailer length [m]
    double trailer_width = 0.8;  // trailer width  [m]
};

/**
 * State of the robot‑trailer system.
 */
struct RobotState {
    double x;            // world x position (m)
    double y;            // world y position (m)
    double theta_robot;  // robot heading (rad)
    double beta;         // articulation angle (rad)
};

/**
 * Core simulator class providing the same methods as the Python version.
 */
class Simulator {
public:
    explicit Simulator(const RobotParameters &params) : params_(params) {}

    /** Convert wheel velocities to forward and angular velocity. */
    std::pair<double, double> wheel_velocities_to_twist(double v_left,
                                                         double v_right) const {
        double v = 0.5 * (v_left + v_right);
        double omega = (v_right - v_left) / params_.wheel_base;
        return {v, omega};
    }

    /** Compute state derivatives for the given control input. */
    std::tuple<double, double, double, double>
    derivatives(const RobotState &state, double v_left, double v_right) const {
        auto [v, omega] = wheel_velocities_to_twist(v_left, v_right);
        double x_dot = v * std::cos(state.theta_robot);
        double y_dot = v * std::sin(state.theta_robot);
        double theta_dot = omega;
        double beta_dot = omega - (v / params_.drawbar_length) * std::sin(state.beta);
        return {x_dot, y_dot, theta_dot, beta_dot};
    }

    /** Single Euler integration step. */
    RobotState step(const RobotState &state, double v_left, double v_right,
                    double dt) const {
        auto [x_dot, y_dot, theta_dot, beta_dot] =
            derivatives(state, v_left, v_right);
        RobotState next;
        next.x = state.x + x_dot * dt;
        next.y = state.y + y_dot * dt;
        next.theta_robot = wrap_angle(state.theta_robot + theta_dot * dt);
        next.beta = wrap_angle(state.beta + beta_dot * dt);
        return next;
    }

    /** Pose of the trailer axle (x, y, heading). */
    std::tuple<double, double, double> trailer_pose(const RobotState &state) const {
        double theta_trailer = state.theta_robot - state.beta;
        double x_trailer = state.x - params_.drawbar_length * std::cos(theta_trailer);
        double y_trailer = state.y - params_.drawbar_length * std::sin(theta_trailer);
        return {x_trailer, y_trailer, theta_trailer};
    }

    /** Simulate a sequence of control inputs. */
    std::vector<RobotState>
    simulate(const RobotState &start,
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

    /** Provide read‑only access to the parameters (mirrors the Python property). */
    const RobotParameters &params() const { return params_; }

private:
    RobotParameters params_;
};

PYBIND11_MODULE(robot_trailer_sim_cpp, m) {
    m.doc() = "C++-accelerated robot‑trailer simulator (pybind11)";

    py::class_<RobotParameters>(m, "RobotParameters")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("wheel_base"), py::arg("drawbar_length"),
             py::arg("robot_length") = 0.8, py::arg("robot_width") = 0.6,
             py::arg("trailer_length") = 1.2, py::arg("trailer_width") = 0.8)
        .def_readwrite("wheel_base", &RobotParameters::wheel_base)
        .def_readwrite("drawbar_length", &RobotParameters::drawbar_length)
        .def_readwrite("robot_length", &RobotParameters::robot_length)
        .def_readwrite("robot_width", &RobotParameters::robot_width)
        .def_readwrite("trailer_length", &RobotParameters::trailer_length)
        .def_readwrite("trailer_width", &RobotParameters::trailer_width);

    py::class_<RobotState>(m, "RobotState")
        .def(py::init<double, double, double, double>(),
             py::arg("x"), py::arg("y"), py::arg("theta_robot"), py::arg("beta"))
        .def_readwrite("x", &RobotState::x)
        .def_readwrite("y", &RobotState::y)
        .def_readwrite("theta_robot", &RobotState::theta_robot)
        .def_readwrite("beta", &RobotState::beta);

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<const RobotParameters &>())
        .def("wheel_velocities_to_twist", &Simulator::wheel_velocities_to_twist)
        .def("derivatives", &Simulator::derivatives)
        .def("step", &Simulator::step)
        .def("trailer_pose", &Simulator::trailer_pose)
        .def("simulate", &Simulator::simulate)
        .def_property_readonly("params", &Simulator::params);
}
