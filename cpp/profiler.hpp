#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>

// Very small profiler that accumulates time (seconds) and call counts per
// named section. Not thread-safe (planner runs single-threaded by default).
struct Profiler {
    struct Entry { double total_seconds = 0.0; std::size_t calls = 0; };
    using Clock = std::chrono::steady_clock;

    void add_sample(const std::string &name, double seconds) {
        auto &e = data[name];
        e.total_seconds += seconds;
        ++e.calls;
    }

    void reset() { data.clear(); }

    std::string summary_string() const {
        // compute overall time if present, otherwise sum entries
        double overall = 0.0;
        auto it = data.find("__total__");
        if (it != data.end()) overall = it->second.total_seconds;
        else {
            for (const auto &p : data) overall += p.second.total_seconds;
        }
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6);
        ss << "Profiler summary (seconds):\n";
        // Create a vector to sort by time
        std::vector<std::pair<std::string, Entry>> items;
        items.reserve(data.size());
        for (const auto &p : data) items.emplace_back(p.first, p.second);
        std::sort(items.begin(), items.end(), [](auto &a, auto &b){ return a.second.total_seconds > b.second.total_seconds; });
        for (const auto &p : items) {
            const auto &name = p.first;
            const auto &e = p.second;
            double pct = overall > 0.0 ? (e.total_seconds / overall * 100.0) : 0.0;
            ss << " - " << std::setw(20) << std::left << name << ": " << std::setw(10) << e.total_seconds
               << "s  (" << std::setw(6) << pct << "%)  calls=" << e.calls << "\n";
        }
        ss << "Overall: " << overall << " s\n";
        return ss.str();
    }

    // Lightweight RAII timer that records to the parent profiler on destruction.
    struct ScopedTimer {
        ScopedTimer(Profiler &p, const std::string &n) : profiler(p), name(n), start(Clock::now()) {}
        ~ScopedTimer() {
            auto end = Clock::now();
            std::chrono::duration<double> dt = end - start;
            profiler.add_sample(name, dt.count());
        }
        Profiler &profiler;
        std::string name;
        Clock::time_point start;
    };

    std::unordered_map<std::string, Entry> data;
};
