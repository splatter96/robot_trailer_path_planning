/**
 * C++ implementation of the Hybrid A* planner.
 *
 * The implementation mirrors the Python ``HybridAStarPlanner`` class but
 * operates entirely in C++ for speed.  It depends on the C++ ``Simulator``
 * class defined in ``cpp/simulator.cpp`` and re‑uses the same data structures
 * (`RobotState`, `RobotParameters`).  The public API is compatible with the
 * Python version so the rest of the codebase can simply import the class from
 * the pybind11 module.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <queue>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace py = pybind11;

// Include the definitions of RobotParameters, RobotState, and Simulator.
#include "simulator.hpp"

// ---------------------------------------------------------------------------
// Helper utilities (same as in the Python implementation)
// ---------------------------------------------------------------------------
static inline double heuristic(const RobotState &s, const RobotState &goal) {
    double dx = s.x - goal.x;
    double dy = s.y - goal.y;
    return std::hypot(dx, dy);
}

// Simple struct to use as a key in unordered_map.
struct StateKey {
    int x, y, theta, beta;
    bool operator==(StateKey const &other) const {
        return x == other.x && y == other.y && theta == other.theta && beta == other.beta;
    }
};

struct StateKeyHash {
    std::size_t operator()(StateKey const &k) const noexcept {
        // Combine hashes of individual fields.
        std::size_t h = std::hash<int>{}(k.x);
        h = h * 31u + std::hash<int>{}(k.y);
        h = h * 31u + std::hash<int>{}(k.theta);
        h = h * 31u + std::hash<int>{}(k.beta);
        return h;
    }
};

static inline StateKey discretize(const RobotState &s, double pos_res = 0.1,
          double ang_res = M_PI / 36.0) { // 5 degrees
    int xi = static_cast<int>(std::round(s.x / pos_res));
    int yi = static_cast<int>(std::round(s.y / pos_res));
    int ti = static_cast<int>(std::round(s.theta_robot / ang_res));
    int bi = static_cast<int>(std::round(s.beta / ang_res));
    return {xi, yi, ti, bi};
}

// Rectangle occupancy test – directly translated from the Numba version.
static bool rectangle_occupied(double cx, double cy, double heading, double length,
                               double width, const py::array_t<bool> &grid,
                               double inv_res, double ox, double oy,
                               int rows, int cols) {
    auto buf = grid.unchecked<2>(); // fast unchecked access
    double hl = length / 2.0;
    double hw = width / 2.0;
    double c = std::cos(heading);
    double s = std::sin(heading);
    for (int dx_sign : {1, -1}) {
        for (int dy_sign : {1, -1}) {
            double dx = hl * dx_sign;
            double dy = hw * dy_sign;
            double x = c * dx - s * dy + cx;
            double y = s * dx + c * dy + cy;
            int ix = static_cast<int>(std::floor((x - ox) * inv_res));
            int iy = static_cast<int>(std::floor((y - oy) * inv_res));
            if (ix < 0 || iy < 0 || ix >= cols || iy >= rows) continue;
            if (buf(iy, ix)) return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Planner class definition
// ---------------------------------------------------------------------------
class HybridAStarPlanner {
public:
    HybridAStarPlanner(const Simulator &sim, const std::vector<std::pair<double, double>> &control_set,
                       const py::array_t<bool> &occupancy_grid = py::array_t<bool>(),
                       double grid_resolution = 0.1,
                       std::pair<double, double> grid_origin = {0.0, 0.0},
                       double dt = 0.1)
        : simulator_(sim), control_set_(control_set), dt_(dt), occupancy_grid_(occupancy_grid),
          grid_resolution_(grid_resolution), ox_(grid_origin.first), oy_(grid_origin.second) {
        if (!occupancy_grid_.is_none()) {
            auto shape = occupancy_grid_.shape();
            rows_ = static_cast<int>(shape[0]);
            cols_ = static_cast<int>(shape[1]);
            inv_res_ = 1.0 / grid_resolution_;
        } else {
            rows_ = cols_ = 0;
            inv_res_ = 0.0;
        }
    }

    std::vector<RobotState> plan(const RobotState &start, const RobotState &goal) {
        using QueueElem = std::tuple<double, double, int, RobotState, StateKey>;
        struct Compare {
            bool operator()(const QueueElem &a, const QueueElem &b) const {
                // std::priority_queue is a max‑heap, so we invert comparison.
                return std::get<0>(a) > std::get<0>(b);
            }
        };
        std::priority_queue<QueueElem, std::vector<QueueElem>, Compare> open_queue;
        int push_counter = 0;
        auto start_key = discretize(start);
        open_queue.emplace(heuristic(start, goal), 0.0, push_counter++, start, start_key);

        // closed map: key -> (g, parent_key, state)
        struct ClosedVal { double g; StateKey parent; RobotState state; };
        std::unordered_map<StateKey, ClosedVal, StateKeyHash> closed;
        closed.emplace(start_key, ClosedVal{0.0, StateKey{-1,-1,-1,-1}, start});

        while (!open_queue.empty()) {
            auto [f, g, tie, current, parent_key] = open_queue.top();
            open_queue.pop();
            auto current_key = discretize(current);

            if (std::hypot(current.x - goal.x, current.y - goal.y) < 0.2) {
                // reconstruct path
                std::vector<RobotState> path;
                auto back_key = current_key;
                while (true) {
                    const auto &val = closed.at(back_key);
                    path.push_back(val.state);
                    if (val.parent == StateKey{-1,-1,-1,-1}) break;
                    back_key = val.parent;
                }
                std::reverse(path.begin(), path.end());
                return path;
            }

            for (const auto &ctrl : control_set_) {
                double v_left = ctrl.first;
                double v_right = ctrl.second;
                RobotState neighbor = simulator_.step(current, v_left, v_right, dt_);
                if (!occupancy_grid_.is_none() && collides(neighbor)) continue;
                auto neighbor_key = discretize(neighbor);
                double tentative_g = g + dt_;
                auto it = closed.find(neighbor_key);
                if (it == closed.end() || tentative_g < it->second.g) {
                    double f_cost = tentative_g + heuristic(neighbor, goal);
                    closed[neighbor_key] = ClosedVal{tentative_g, current_key, neighbor};
                    open_queue.emplace(f_cost, tentative_g, push_counter++, neighbor, current_key);
                    ++push_counter;
                }
            }
        }
        return {};
    }

private:
    const Simulator &simulator_;
    std::vector<std::pair<double, double>> control_set_;
    double dt_;
    py::array_t<bool> occupancy_grid_; // may be None
    double grid_resolution_;
    double ox_, oy_;
    double inv_res_ = 0.0;
    int rows_ = 0, cols_ = 0;

    bool collides(const RobotState &state) const {
        // Robot chassis
        if (rectangle_occupied(state.x, state.y, state.theta_robot,
                               simulator_.params().robot_length,
                               simulator_.params().robot_width,
                               occupancy_grid_, inv_res_, ox_, oy_, rows_, cols_))
            return true;
        // Trailer
        auto [tx, ty, ttheta] = simulator_.trailer_pose(state);
        if (rectangle_occupied(tx, ty, ttheta,
                               simulator_.params().trailer_length,
                               simulator_.params().trailer_width,
                               occupancy_grid_, inv_res_, ox_, oy_, rows_, cols_))
            return true;
        return false;
    }
};

PYBIND11_MODULE(robot_trailer_cpp, m) {
    m.doc() = "C++ implementation of robot‑trailer simulator and Hybrid A* planner";

    // Re‑export the simulator bindings (they are defined in simulator.cpp).
    // The simulator.cpp file already creates a module named
    // ``robot_trailer_sim_cpp``.  Here we expose the same classes under this
    // combined module for convenience.
    // Note: pybind11 will merge the definitions when the same name is used.
    // Import the existing definitions.
    py::module::import("robot_trailer_sim_cpp");

    py::class_<HybridAStarPlanner>(m, "HybridAStarPlanner")
        .def(py::init<const Simulator &, const std::vector<std::pair<double, double>> &,
                      const py::array_t<bool> &, double, std::pair<double, double>, double>(),
             py::arg("simulator"), py::arg("control_set"), py::arg("occupancy_grid") = py::none(),
             py::arg("grid_resolution") = 0.1, py::arg("grid_origin") = std::make_pair(0.0, 0.0),
             py::arg("dt") = 0.1)
        .def("plan", &HybridAStarPlanner::plan);
}
