#pragma once

#include <chrono>
#include <iostream>

namespace lmj {
using namespace std::chrono;

struct timer {
    time_point <high_resolution_clock> start_time = high_resolution_clock::now();

    bool print = true;

    timer() = default;

    explicit timer(bool p) : print{p} {}

    [[nodiscard]] auto curr_time() const {
        auto now = high_resolution_clock::now();
        auto dur = duration_cast<nanoseconds>(now - start_time);
        return static_cast<double>(dur.count()) / 1e9;
    }

    [[nodiscard]] auto elapsed() const {
        return curr_time();
    }

    ~timer() {
        if (print) {
            std::cerr << curr_time() << "s\n";
        }
    }
};
}