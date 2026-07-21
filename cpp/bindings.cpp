#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "simulator.hpp"
#include "planner.hpp"

namespace py = pybind11;

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
        .def_readwrite("beta", &RobotState::beta)
        .def_readwrite("trailer_x", &RobotState::trailer_x)
        .def_readwrite("trailer_y", &RobotState::trailer_y)
        .def_readwrite("theta_trailer", &RobotState::theta_trailer);

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<const RobotParameters &>())
        .def("wheel_velocities_to_twist", &Simulator::wheel_velocities_to_twist)
        .def("trailer_pose", &Simulator::trailer_pose)
        .def("simulate", &Simulator::simulate)
        .def_property_readonly("params", &Simulator::params);
}

PYBIND11_MODULE(robot_trailer_cpp, m) {
    m.doc() = "C++-accelerated planner bindings";

    // Ensure simulator bindings are available (they may be in the other module).
    py::module::import("robot_trailer_sim_cpp");

    py::class_<HybridAStarPlanner>(m, "HybridAStarPlanner")
        .def(py::init([](const Simulator &sim, const std::vector<std::pair<double,double>> &control_set,
                         py::array_t<bool> occupancy_grid, double grid_resolution,
                         std::pair<double,double> grid_origin, double dt) {
            std::vector<uint8_t> occ_vec;
            int rows = 0, cols = 0;
            if (!occupancy_grid.is_none()) {
                auto buf = occupancy_grid.unchecked<2>();
                rows = static_cast<int>(occupancy_grid.shape(0));
                cols = static_cast<int>(occupancy_grid.shape(1));
                occ_vec.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        occ_vec[i * cols + j] = buf(i, j) ? 1 : 0;
                    }
                }
            }
            return HybridAStarPlanner(sim, control_set, occ_vec, rows, cols, grid_resolution, grid_origin, dt);
        }),
             py::arg("simulator"), py::arg("control_set"), py::arg("occupancy_grid") = py::none(),
             py::arg("grid_resolution") = 0.1, py::arg("grid_origin") = std::make_pair(0.0, 0.0),
             py::arg("dt") = 0.1)
        .def("plan", &HybridAStarPlanner::plan)
        .def("get_explored", &HybridAStarPlanner::get_explored)
        .def("profile_summary", &HybridAStarPlanner::profile_summary)
        .def("reverse_distance_map", [](const HybridAStarPlanner &p, const RobotState &goal) {
            auto vec = p.reverse_distance_map(goal);
            if (vec.empty()) return py::array_t<double>();
            int rows = p.rows();
            int cols = p.cols();
            py::array_t<double> arr({rows, cols});
            auto out = arr.mutable_unchecked<2>();
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    out(i, j) = vec[i * cols + j];
            return arr;
        }, py::arg("goal"));
}
