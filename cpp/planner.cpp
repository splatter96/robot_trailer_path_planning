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

#include <queue>
#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <algorithm>
#include <iostream>

#include "planner.hpp"

// Footprint lookup for fast collision checking
// #include "footprint_lookup.hpp"

// Include the definitions of RobotParameters, RobotState, and Simulator.
#include "simulator.hpp"
// Lightweight profiler for timing hot regions.
#include "profiler.hpp"

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
static std::vector<double> compute_reverse_dist(const std::vector<uint8_t> &occupancy_grid,
                                                int rows, int cols,
                                                double grid_resolution,
                                                const std::pair<double, double> &grid_origin,
                                                const RobotState &goal) {
    std::vector<double> dist_arr(rows * cols, INF_COST);
    double ox = grid_origin.first;
    double oy = grid_origin.second;
    double inv_res = 1.0 / grid_resolution;

    // Convert goal world coordinates to grid indices.
    int gx = static_cast<int>(std::floor((goal.x - ox) * inv_res));
    int gy = static_cast<int>(std::floor((goal.y - oy) * inv_res));
    if (gx >= 0 && gy >= 0 && gx < cols && gy < rows && !occupancy_grid[gy * cols + gx]) {
        dist_arr[gy * cols + gx] = 0.0;
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
        if (c > dist_arr[y * cols + x]) continue; // stale entry
        for (int k = 0; k < 8; ++k) {
            int ny = y + dY[k];
            int nx = x + dX[k];
            if (ny < 0 || nx < 0 || ny >= rows || nx >= cols) continue;
            if (occupancy_grid[ny * cols + nx]) continue; // occupied cells stay INF
            double nd = c + cost[k];
            if (nd < dist_arr[ny * cols + nx]) {
                dist_arr[ny * cols + nx] = nd;
                pq.emplace(nd, ny, nx);
            }
        }
    }
    return dist_arr;
}

// Heuristic that first tries the reverse distance map; falls back to Euclidean.
static inline double heuristic(const RobotState &s, const RobotState &goal,
                               const std::vector<double> *rev_dist,
                               int rev_rows, int rev_cols,
                               double grid_resolution,
                               const std::pair<double, double> &grid_origin) {
    if (rev_dist && !rev_dist->empty()) {
        double ox = grid_origin.first;
        double oy = grid_origin.second;
        double inv_res = 1.0 / grid_resolution;
        int ix = static_cast<int>(std::floor((s.x - ox) * inv_res));
        int iy = static_cast<int>(std::floor((s.y - oy) * inv_res));
        if (ix >= 0 && iy >= 0 && ix < rev_cols && iy < rev_rows) {
            double d = (*rev_dist)[iy * rev_cols + ix];
            if (d < INF_COST) return d;
        }
    }
    // Fallback Euclidean distance.
    double dx = s.x - goal.x;
    double dy = s.y - goal.y;
    return std::hypot(dx, dy);
}

struct StateKey {
    int x, y, theta, beta;
    bool operator==(StateKey const &other) const {
        return x == other.x && y == other.y && theta == other.theta && beta == other.beta;
    }
};

// ---------------------------------------------------------------------------
// Internal data structures for the A* search.
// ---------------------------------------------------------------------------
// Closed list entry storing cost information and the associated state.
struct ClosedVal {
    double g;          // Cost from start to this node.
    double f;          // g + heuristic.
    int parent;        // Index of parent node in the nodes vector.
    RobotState state;  // The robot state at this node.
    int heap_index;    // Position in the binary heap (or -1 if not in heap).
    bool expanded;     // Whether the node has been expanded.
};

// Simple binary min‑heap for the open list. It operates on indices into the
// `nodes` vector and uses the `ClosedVal::f` (and then `g`) for ordering.
struct OpenHeap {
    std::vector<int> heap;                 // Heap of node indices.
    std::vector<ClosedVal> *nodes;         // Back‑reference to the node storage.

    // `reserve_size` allows the caller to pre‑allocate the heap capacity.
    explicit OpenHeap(std::vector<ClosedVal> *nodes_ptr, size_t reserve_size = 0)
        : nodes(nodes_ptr) {
        if (reserve_size) heap.reserve(reserve_size);
    }

    bool empty() const { return heap.empty(); }

    void swap_positions(int i, int j) {
        int a = heap[i];
        int b = heap[j];
        heap[i] = b;
        heap[j] = a;
        (*nodes)[a].heap_index = j;
        (*nodes)[b].heap_index = i;
    }

    bool compare(int i, int j) const {
        const ClosedVal &a = (*nodes)[heap[i]];
        const ClosedVal &b = (*nodes)[heap[j]];
        if (a.f != b.f) return a.f < b.f;
        return a.g < b.g;
    }

    void sift_up(int pos) {
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (compare(pos, parent)) {
                swap_positions(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    void sift_down(int pos) {
        int n = static_cast<int>(heap.size());
        while (true) {
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            int smallest = pos;
            if (left < n && compare(left, smallest)) smallest = left;
            if (right < n && compare(right, smallest)) smallest = right;
            if (smallest == pos) break;
            swap_positions(pos, smallest);
            pos = smallest;
        }
    }

    void push(int node_index) {
        heap.push_back(node_index);
        (*nodes)[node_index].heap_index = static_cast<int>(heap.size() - 1);
        sift_up(static_cast<int>(heap.size() - 1));
    }

    int pop_top() {
        int top = heap[0];
        int last = heap.back();
        heap.pop_back();
        if (!heap.empty()) {
            heap[0] = last;
            (*nodes)[last].heap_index = 0;
            sift_down(0);
        }
        (*nodes)[top].heap_index = -1;
        return top;
    }

    void decrease_key(int node_index) {
        int pos = (*nodes)[node_index].heap_index;
        if (pos >= 0) sift_up(pos);
    }
};

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t pack_state_key(const StateKey &k) {
    uint64_t x = static_cast<uint32_t>(k.x);
    uint64_t y = static_cast<uint32_t>(k.y);
    uint64_t theta = static_cast<uint32_t>(k.theta & 0xffff);
    uint64_t beta = static_cast<uint32_t>(k.beta & 0xffff);
    return (x << 32) | (y << 16) | (theta << 8) | beta;
}

struct StateIndexMap {
    struct Entry { uint64_t key = 0; int node_idx = -1; };
    std::vector<Entry> entries;
    size_t mask = 0;
    int count = 0;

    explicit StateIndexMap(size_t capacity) {
        size_t size = 1;
        while (size < capacity * 2) size <<= 1;
        entries.assign(size, Entry{});
        mask = size - 1;
        count = 0;
    }

    int find(const StateKey &key) const {
        uint64_t packed = pack_state_key(key);
        size_t pos = splitmix64(packed) & mask;
        while (true) {
            const Entry &entry = entries[pos];
            if (entry.node_idx < 0) return -1;
            if (entry.key == packed) return entry.node_idx;
            pos = (pos + 1) & mask;
        }
    }

    void insert(const StateKey &key, int node_idx) {
        if (count * 2 >= static_cast<int>(entries.size())) resize(entries.size() * 2);
        uint64_t packed = pack_state_key(key);
        size_t pos = splitmix64(packed) & mask;
        while (true) {
            Entry &entry = entries[pos];
            if (entry.node_idx < 0) {
                entry.key = packed;
                entry.node_idx = node_idx;
                ++count;
                return;
            }
            if (entry.key == packed) {
                entry.node_idx = node_idx;
                return;
            }
            pos = (pos + 1) & mask;
        }
    }

    void resize(size_t new_size) {
        std::vector<Entry> old_entries = std::move(entries);
        entries.assign(new_size, Entry{});
        mask = new_size - 1;
        count = 0;
        for (const Entry &entry : old_entries) {
            if (entry.node_idx >= 0) {
                size_t pos = splitmix64(entry.key) & mask;
                while (entries[pos].node_idx >= 0) {
                    pos = (pos + 1) & mask;
                }
                entries[pos] = entry;
                ++count;
            }
        }
    }
};

// Produce a StateKey using the cached discretized values stored in the state.
// If the indices have not been initialised (negative), fall back to the
// original rounding calculation to preserve correctness.
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
                               double width, const std::vector<uint8_t> &grid,
                               double inv_res, double ox, double oy,
                               int rows, int cols) {
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
            if (grid[iy * cols + ix]) return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Planner class definition
// ---------------------------------------------------------------------------
// Implementations of HybridAStarPlanner methods (defined in planner.hpp)

HybridAStarPlanner::HybridAStarPlanner(const Simulator &sim, const std::vector<std::pair<double, double>> &control_set,
                      const std::vector<uint8_t> &occupancy_grid,
                      int rows, int cols,
                      double grid_resolution,
                      std::pair<double, double> grid_origin,
                      double dt)
    : simulator_(sim), control_set_(control_set), dt_(dt), occupancy_grid_(occupancy_grid),
      grid_resolution_(grid_resolution), ox_(grid_origin.first), oy_(grid_origin.second),
      rows_(rows), cols_(cols) {

    if (rows_ > 0 && cols_ > 0 && occupancy_grid_.size() == static_cast<size_t>(rows_ * cols_)) {
        inv_res_ = 1.0 / grid_resolution_;
    } else {
        rows_ = cols_ = 0;
        inv_res_ = 0.0;
        occupancy_grid_.clear();
    }

    // robot_lookup_ = std::make_unique<FootprintLookup>(
    //     simulator_.params().robot_length,
    //     simulator_.params().robot_width,
    //     grid_resolution_);
    // trailer_lookup_ = std::make_unique<FootprintLookup>(
    //     simulator_.params().trailer_length,
    //     simulator_.params().trailer_width,
    //     grid_resolution_);

    clearance_threshold_ = std::max({simulator_.params().robot_length,
                                     simulator_.params().robot_width,
                                     simulator_.params().trailer_length,
                                     simulator_.params().trailer_width});

    // Initialise the simulator's derivative cache for the known control set.
    // This allows the planner to use a fast lookup during the search.
    simulator_.init_derivative_cache(control_set_);
    if (rows_ > 0 && cols_ > 0) {
        distance_grid_.assign(rows_ * cols_, INF_COST);
        std::queue<std::pair<int, int>> q;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (occupancy_grid_[i * cols_ + j]) {
                    distance_grid_[i * cols_ + j] = 0.0;
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
            double cur = distance_grid_[ci * cols_ + cj];
            for (int k = 0; k < 8; ++k) {
                int ni = ci + di[k];
                int nj = cj + dj[k];
                if (ni < 0 || nj < 0 || ni >= rows_ || nj >= cols_) continue;
                double nd = cur + cost[k];
                if (nd < distance_grid_[ni * cols_ + nj]) {
                    distance_grid_[ni * cols_ + nj] = nd;
                    q.emplace(ni, nj);
                }
            }
        }
    }
    std::cout << "HybridAStarPlanner initialized with grid resolution " << grid_resolution_
              << ", occupancy grid size " << rows_ << "x" << cols_ << std::endl;
}

void HybridAStarPlanner::reset_explored() { explored_.clear(); }

std::vector<RobotState> HybridAStarPlanner::get_explored() const { return explored_; }

std::vector<double> HybridAStarPlanner::reverse_distance_map(const RobotState &goal) const {
    if (rows_ == 0 || cols_ == 0) return std::vector<double>();
    return compute_reverse_dist(occupancy_grid_, rows_, cols_, grid_resolution_, {ox_, oy_}, goal);
}

std::vector<RobotState> HybridAStarPlanner::plan(const RobotState &start, const RobotState &goal) {
    // ---------------------------------------------------------------------
    // Initialise search state
    // ---------------------------------------------------------------------
    explored_.clear();               // Reset the list of explored states.
    profiler_.reset();               // Clear any previous profiling data.
    const std::size_t open_reserve = 262144; // Pre‑allocate space for the heap (tuneable).

    // Discretise the start state to obtain a unique key for the closed list.
    auto start_key = discretize(start);

    // ---------------------------------------------------------------------
    // Optional reverse‑distance heuristic preparation
    // ---------------------------------------------------------------------
    std::vector<double> rev_dist;               // Cached reverse distance map.
    const std::vector<double>* rev_dist_ptr = nullptr;
    if (rows_ > 0 && cols_ > 0) {
        // Compute a distance‑to‑goal field on the occupancy grid.
        rev_dist = compute_reverse_dist(occupancy_grid_, rows_, cols_, grid_resolution_, {ox_, oy_}, goal);
        rev_dist_ptr = &rev_dist;
    }
    const bool has_occupancy = (rows_ > 0 && cols_ > 0);

    // ---------------------------------------------------------------------
    // Data structures for the A* algorithm
    // ---------------------------------------------------------------------
    std::vector<ClosedVal> nodes;               // Closed list entries.
    nodes.reserve(32768);
    StateIndexMap state_index(1 << 22);         // Fast lookup from discretised state to node index.
    OpenHeap open_list(&nodes, open_reserve);   // Open list (binary min‑heap).

    // Initialise the start node and push it onto the open list.
    double start_f = heuristic(start, goal, rev_dist_ptr, rows_, cols_, grid_resolution_, {ox_, oy_});
    nodes.push_back(ClosedVal{0.0, start_f, -1, start, -1, false});
    state_index.insert(start_key, 0);
    open_list.push(0);

    const double goal_thresh_sqr = 0.2 * 0.2; // Goal proximity threshold (squared meters).

    // ---------------------------------------------------------------------
    // Main A* loop – expand the most promising node until the goal is reached
    // or the open list is exhausted.
    // ---------------------------------------------------------------------
    while (!open_list.empty()) {
        // Pop the node with the lowest f‑cost.
        int current_idx = open_list.pop_top();
        ClosedVal &current_node = nodes[current_idx];
        current_node.expanded = true;          // Mark as expanded to avoid re‑expansion.
        RobotState current = current_node.state;
        double g = current_node.g;             // Cost from start to this node.
        explored_.push_back(current);

        // Goal test – if within the threshold, reconstruct the path.
        if ((current.x - goal.x) * (current.x - goal.x) +
            (current.y - goal.y) * (current.y - goal.y) < goal_thresh_sqr) {
            std::vector<RobotState> path;
            int back_idx = current_idx;
            while (back_idx >= 0) {
                path.push_back(nodes[back_idx].state);
                back_idx = nodes[back_idx].parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Expand neighbours by applying each control primitive.
        for (size_t ctrl_idx = 0; ctrl_idx < control_set_.size(); ++ctrl_idx) {
            // Use the cached derivative step for the given control index.
            RobotState neighbor = simulator_.step_cached(current, ctrl_idx, dt_);
            // Discard states that collide with obstacles (if an occupancy grid is present).
            if (has_occupancy && collides(neighbor)) continue;

            // Discretise the neighbour to obtain a lookup key.
            auto neighbor_key = discretize(neighbor);
            double tentative_g = g + dt_; // Cost to reach the neighbour.
            int existing_idx = state_index.find(neighbor_key);

            if (existing_idx < 0) {
                // New state – compute f‑cost and add to both closed list and open list.
                double f_cost = tentative_g + heuristic(neighbor, goal, rev_dist_ptr, rows_, cols_, grid_resolution_, {ox_, oy_});
                int neighbor_idx = static_cast<int>(nodes.size());
                nodes.push_back(ClosedVal{tentative_g, f_cost, current_idx, neighbor, -1, false});
                state_index.insert(neighbor_key, neighbor_idx);
                open_list.push(neighbor_idx);
            } else {
                // Existing state – possibly improve its cost.
                int neighbor_idx = existing_idx;
                ClosedVal &neighbor_node = nodes[neighbor_idx];
                if (tentative_g < neighbor_node.g) {
                    double f_cost = tentative_g + heuristic(neighbor, goal, rev_dist_ptr, rows_, cols_, grid_resolution_, {ox_, oy_});
                    neighbor_node.g = tentative_g;
                    neighbor_node.f = f_cost;
                    neighbor_node.parent = current_idx;
                    neighbor_node.state = neighbor;
                    // Update the open list position based on the node's current state.
                    if (neighbor_node.expanded) {
                        neighbor_node.expanded = false;
                        open_list.push(neighbor_idx);
                    } else if (neighbor_node.heap_index >= 0) {
                        open_list.decrease_key(neighbor_idx);
                    } else {
                        open_list.push(neighbor_idx);
                    }
                }
            }
        }
    }
    // No path found – return an empty vector.
    return {};
}

// Return a human-readable profiler summary string.
std::string HybridAStarPlanner::profile_summary() const { return profiler_.summary_string(); }

int HybridAStarPlanner::rows() const { return rows_; }
int HybridAStarPlanner::cols() const { return cols_; }

bool HybridAStarPlanner::collides(const RobotState &state) const {
    if (rows_ == 0 || cols_ == 0) return false;
    // Use the pre‑computed distance map. Cache grid indices inside the state
    // to avoid recomputing floor() operations for every collision check.
    // The indices are mutable, so they can be updated even though `state` is
    // passed as a const reference.

    // ----- Robot centre -----
    if (state.ix < 0 || state.iy < 0) {
        state.ix = static_cast<int>(std::floor((state.x - ox_) * inv_res_));
        state.iy = static_cast<int>(std::floor((state.y - oy_) * inv_res_));
    }
    if (state.ix >= 0 && state.iy >= 0 && state.ix < cols_ && state.iy < rows_) {
        if (distance_grid_[state.iy * cols_ + state.ix] < clearance_threshold_) return true;
    }

    // ----- Trailer centre -----
    if (state.ix_t < 0 || state.iy_t < 0) {
        state.ix_t = static_cast<int>(std::floor((state.trailer_x - ox_) * inv_res_));
        state.iy_t = static_cast<int>(std::floor((state.trailer_y - oy_) * inv_res_));
    }
    if (state.ix_t >= 0 && state.iy_t >= 0 && state.ix_t < cols_ && state.iy_t < rows_) {
        if (distance_grid_[state.iy_t * cols_ + state.ix_t] < clearance_threshold_) return true;
    }
    return false;
}

// Bindings moved to cpp/bindings.cpp
