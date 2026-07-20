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
#include <memory>
#include <algorithm>
#include <iostream>

// Footprint lookup for fast collision checking
#include "footprint_lookup.hpp"

namespace py = pybind11;

// Include the definitions of RobotParameters, RobotState, and Simulator.
#include "simulator.hpp"

// ---------------------------------------------------------------------------
// Helper utilities (same as in the Python implementation)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Reverse‑search (cheap A*) heuristic utilities
// ---------------------------------------------------------------------------
static const double INF_COST = 1e12;

// Compute a grid‑based distance‑to‑goal map using an 8‑connected Dijkstra.
// The map stores the shortest Euclidean distance (in metres) from each free
// cell to the goal, ignoring the robot's dynamics.  Occupied cells retain
// INF_COST.
static py::array_t<double> compute_reverse_dist(const py::array_t<bool> &occupancy_grid,
                                                double grid_resolution,
                                                const std::pair<double, double> &grid_origin,
                                                const RobotState &goal) {
    auto shape = occupancy_grid.shape();
    int rows = static_cast<int>(shape[0]);
    int cols = static_cast<int>(shape[1]);
    double ox = grid_origin.first;
    double oy = grid_origin.second;
    double inv_res = 1.0 / grid_resolution;

    py::array_t<double> dist_arr({rows, cols});
    auto dist = dist_arr.mutable_unchecked<2>();
    auto occ = occupancy_grid.unchecked<2>();

    // Initialise all distances to INF, then set goal cell to zero.
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dist(i, j) = INF_COST;
        }
    }
    // Convert goal world coordinates to grid indices.
    int gx = static_cast<int>(std::floor((goal.x - ox) * inv_res));
    int gy = static_cast<int>(std::floor((goal.y - oy) * inv_res));
    if (gx >= 0 && gy >= 0 && gx < cols && gy < rows && !occ(gy, gx)) {
        dist(gy, gx) = 0.0;
    } else {
        // Goal inside obstacle or out of bounds – return all INF.
        return dist_arr;
    }

    using Node = std::tuple<double, int, int>; // (cost, y, x)
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.emplace(0.0, gy, gx);

    const double diag = std::sqrt(2.0) * grid_resolution;
    const int dY[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dX[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    const double cost[8] = {grid_resolution, grid_resolution, grid_resolution, grid_resolution,
                             diag, diag, diag, diag};

    while (!pq.empty()) {
        auto [c, y, x] = pq.top();
        pq.pop();
        if (c > dist(y, x)) continue; // stale entry
        for (int k = 0; k < 8; ++k) {
            int ny = y + dY[k];
            int nx = x + dX[k];
            if (ny < 0 || nx < 0 || ny >= rows || nx >= cols) continue;
            if (occ(ny, nx)) continue; // occupied cells stay INF
            double nd = c + cost[k];
            if (nd < dist(ny, nx)) {
                dist(ny, nx) = nd;
                pq.emplace(nd, ny, nx);
            }
        }
    }
    return dist_arr;
}

// Heuristic that first tries the reverse distance map; falls back to Euclidean.
static inline double heuristic(const RobotState &s, const RobotState &goal,
                               const py::array_t<double> *rev_dist,
                               double grid_resolution,
                               const std::pair<double, double> &grid_origin) {
    if (rev_dist && rev_dist->size() > 0) {
        auto shape = rev_dist->shape();
        int rows = static_cast<int>(shape[0]);
        int cols = static_cast<int>(shape[1]);
        double ox = grid_origin.first;
        double oy = grid_origin.second;
        double inv_res = 1.0 / grid_resolution;
        int ix = static_cast<int>(std::floor((s.x - ox) * inv_res));
        int iy = static_cast<int>(std::floor((s.y - oy) * inv_res));
        if (ix >= 0 && iy >= 0 && ix < cols && iy < rows) {
            auto dist = rev_dist->unchecked<2>();
            double d = dist(iy, ix);
            if (d < INF_COST) return d;
        }
    }
    // Fallback Euclidean distance.
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

            // Initialise footprint lookups for robot and trailer (kept for possible future use).
            robot_lookup_ = std::make_unique<FootprintLookup>(
                simulator_.params().robot_length,
                simulator_.params().robot_width,
                grid_resolution_);
            trailer_lookup_ = std::make_unique<FootprintLookup>(
                simulator_.params().trailer_length,
                simulator_.params().trailer_width,
                grid_resolution_);

            // Compute a simple clearance threshold: the longest edge among robot and trailer.
            clearance_threshold_ = std::max({simulator_.params().robot_length,
                                             simulator_.params().robot_width,
                                             simulator_.params().trailer_length,
                                             simulator_.params().trailer_width});

            // ------------------------------------------------------------
            // Build a distance map (Euclidean distance to the nearest obstacle).
            // The map is stored as a double array of the same dimensions as the
            // occupancy grid.  Cells that are occupied get distance 0, free cells
            // get the Euclidean distance (in metres) to the nearest occupied cell.
            // ------------------------------------------------------------
            if (!occupancy_grid_.is_none()) {
                // Create mutable array for distances.
                distance_grid_ = py::array_t<double>({rows_, cols_});
                auto occ = occupancy_grid_.unchecked<2>();
                auto dist = distance_grid_.mutable_unchecked<2>();
                const double INF = 1e12;
                // Initialise distances.
                for (int i = 0; i < rows_; ++i) {
                    for (int j = 0; j < cols_; ++j) {
                        dist(i, j) = occ(i, j) ? 0.0 : INF;
                    }
                }
                // Multi‑source BFS (8‑connected) to propagate Euclidean distances.
                std::queue<std::pair<int, int>> q;
                for (int i = 0; i < rows_; ++i) {
                    for (int j = 0; j < cols_; ++j) {
                        if (occ(i, j)) {
                            q.emplace(i, j);
                        }
                    }
                }
                const double res = grid_resolution_;
                const double diag = std::sqrt(2.0) * res;
                const int di[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
                const int dj[8] = {0, 0, -1, 1, -1, 1, -1, 1};
                const double cost[8] = {res, res, res, res, diag, diag, diag, diag};
                while (!q.empty()) {
                    auto [ci, cj] = q.front();
                    q.pop();
                    double cur = dist(ci, cj);
                    for (int k = 0; k < 8; ++k) {
                        int ni = ci + di[k];
                        int nj = cj + dj[k];
                        if (ni < 0 || nj < 0 || ni >= rows_ || nj >= cols_) continue;
                        double nd = cur + cost[k];
                        if (nd < dist(ni, nj)) {
                            dist(ni, nj) = nd;
                            q.emplace(ni, nj);
                        }
                    }
                }
            }
        std::cout << "HybridAStarPlanner initialized with grid resolution " << grid_resolution_
                  << ", occupancy grid size " << rows_ << "x" << cols_ << std::endl;
    }

    // Reset explored nodes before each planning run.
    void reset_explored() { explored_.clear(); }

    // Retrieve the list of nodes that were expanded during the last plan.
    std::vector<RobotState> get_explored() const { return explored_; }

    // Return the reverse distance‑to‑goal map for a given goal state.  This can be
    // visualised in Python to see how the heuristic guides the planner.
    py::array_t<double> reverse_distance_map(const RobotState &goal) const {
        if (occupancy_grid_.is_none()) {
            // Return an empty array if no occupancy grid is present.
            return py::array_t<double>();
        }
        return compute_reverse_dist(occupancy_grid_, grid_resolution_, {ox_, oy_}, goal);
    }

    std::vector<RobotState> plan(const RobotState &start, const RobotState &goal) {
        // Ensure the container is empty for a fresh run.
        explored_.clear();
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
        // Compute reverse distance map if we have an occupancy grid.
        py::array_t<double> rev_dist;
        const py::array_t<double>* rev_dist_ptr = nullptr;
        if (!occupancy_grid_.is_none()) {
            rev_dist = compute_reverse_dist(occupancy_grid_, grid_resolution_, {ox_, oy_}, goal);
            rev_dist_ptr = &rev_dist;
        }
        open_queue.emplace(heuristic(start, goal, rev_dist_ptr, grid_resolution_, {ox_, oy_}),
                           0.0, push_counter++, start, start_key);
        // open_queue.emplace(heuristic(start, goal), 0.0, push_counter++, start, start_key);

        // closed map: key -> (g, parent_key, state)
        struct ClosedVal { double g; StateKey parent; RobotState state; };
        std::unordered_map<StateKey, ClosedVal, StateKeyHash> closed;
        closed.emplace(start_key, ClosedVal{0.0, StateKey{-1,-1,-1,-1}, start});

        while (!open_queue.empty()) {
            auto [f, g, tie, current, parent_key] = open_queue.top();
            open_queue.pop();
            // Record every node that is expanded (popped from the queue).
            explored_.push_back(current);
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
                    // double f_cost = tentative_g + heuristic(neighbor, goal);
                    double f_cost = tentative_g + heuristic(neighbor, goal, rev_dist_ptr, grid_resolution_, {ox_, oy_});
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
    std::vector<RobotState> explored_;
    py::array_t<bool> occupancy_grid_; // may be None
    double grid_resolution_;
    double ox_, oy_;
    double inv_res_ = 0.0;
    int rows_ = 0, cols_ = 0;

    // Fast lookup tables for robot and trailer footprints (kept for possible future extensions).
    std::unique_ptr<FootprintLookup> robot_lookup_;
    std::unique_ptr<FootprintLookup> trailer_lookup_;

    // Distance map (Euclidean distance to nearest obstacle) and clearance threshold.
    py::array_t<double> distance_grid_; // same dimensions as occupancy_grid_
    double clearance_threshold_ = 0.0;

    bool collides(const RobotState &state) const {
        if (occupancy_grid_.is_none()) return false;
        // Use the pre‑computed distance map.  If the distance from the centre of
        // the robot (or trailer) to the nearest obstacle is smaller than the
        // longest edge length, we consider it a collision.
        auto dist = distance_grid_.unchecked<2>();

        // ----- Robot centre -----
        int ix = static_cast<int>(std::floor((state.x - ox_) * inv_res_));
        int iy = static_cast<int>(std::floor((state.y - oy_) * inv_res_));
        if (ix >= 0 && iy >= 0 && ix < cols_ && iy < rows_) {
            if (dist(iy, ix) < clearance_threshold_) return true;
        }

        // ----- Trailer centre -----
        // Use the cached trailer pose stored in the RobotState.
        int ix_t = static_cast<int>(std::floor((state.trailer_x - ox_) * inv_res_));
        int iy_t = static_cast<int>(std::floor((state.trailer_y - oy_) * inv_res_));
        if (ix_t >= 0 && iy_t >= 0 && ix_t < cols_ && iy_t < rows_) {
            if (dist(iy_t, ix_t) < clearance_threshold_) return true;
        }
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
        .def("plan", &HybridAStarPlanner::plan)
        .def("get_explored", &HybridAStarPlanner::get_explored)
        .def("reverse_distance_map", &HybridAStarPlanner::reverse_distance_map,
             py::arg("goal"))
        ;
}
