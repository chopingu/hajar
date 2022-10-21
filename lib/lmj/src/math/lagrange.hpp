#pragma once

#include <cstdint>
#include <utility>
#include <array>

#include "../utils/utils.hpp"
#include "../containers/containers.hpp"

namespace lmj::lagrange {
namespace detail {
template<std::size_t length, class...Args>
constexpr auto format_helper(static_vector<point, length> &arr, long double a, long double b, Args &&...pack) {
    arr.push_back(point{a, b});
    if constexpr (sizeof...(pack)) {
        format_helper(arr, std::forward<Args>(pack)...);
    }
}

template<class... Args>
constexpr auto data_format(Args &&...pack) {
    static_assert(sizeof...(Args) % 2 == 0, "even number of arguments required, interpreted as pairs of (x, y)");
    constexpr auto num_pairs = sizeof...(Args) / 2;
    static_vector<point, num_pairs> v;
    format_helper(v, pack...);
    std::array<point, num_pairs> res{};
    std::copy(v.begin(), v.end(), res.begin());
    return res;
}
}

template<std::size_t length>
constexpr long double interpolate(long double x, std::array<point, length> const &points) {
    long double result = 0;
    for (std::size_t i = 0; i < length; ++i) {
        long double p = points[i].y;
        for (std::size_t j = 0; j < length; ++j) {
            if (i == j)
                continue;
            p *= (x - points[j].x) / (points[i].x - points[j].x);
        }
        result += p;
    }
    return result;
}

/**
 * @brief get polynomial function takes on values of all pairs (x, y) specified in parameter points
 * @note uses lagrange interpolation
 * @param points
 * @return
 */
constexpr auto get_function(numbers auto &&... points) {
    return [=](long double x) { return interpolate(x, detail::data_format(points...)); };
}

static_assert(lmj::lagrange::get_function(0, 0, 0.5, 0.25, 1, 1)(0) == 0); // y = x^2
static_assert(lmj::lagrange::get_function(0, 0, 0.5, 0.25, 1, 1)(1) == 1); // y = x^2
static_assert(lmj::lagrange::get_function(0, 0, 0.5, 0.25, 1, 1)(2) == 4); // y = x^2
static_assert(lmj::lagrange::get_function(0, 0, 0.5, 0.25, 1, 1)(3) == 9); // y = x^2
static_assert(lmj::lagrange::get_function(0, 0, 2, 3, 5, 20)(3) == 7);
}