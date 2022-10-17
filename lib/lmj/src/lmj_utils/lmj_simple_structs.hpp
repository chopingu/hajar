#pragma once

namespace lmj {
struct point {
    long double x{}, y{};

    constexpr point() = default;

    template<class T, class G>
    constexpr point(T &&x, G &&y) : x(x), y(y) {}
};
}