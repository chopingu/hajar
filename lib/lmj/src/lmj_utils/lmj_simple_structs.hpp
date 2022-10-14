#pragma once

namespace lmj {
struct point {
    long double x{}, y{};

    constexpr point() = default;

    template<class T, class G>
    constexpr point(T &&_x, G &&_y) : x(_x), y(_y) {}
};
}