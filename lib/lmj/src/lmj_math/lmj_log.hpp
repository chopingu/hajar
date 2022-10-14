#pragma once

#include <limits>
#include "lmj_misc_math.hpp"
#include "lmj_newton_raphson.hpp"

namespace lmj {
constexpr auto ln_cp_impl(long double n) {
    constexpr long double ln_1001 = 0.000999500333083; // ln(1.001)

    struct approximation {
        long double value, exponent;
    };

    if (n == 1.001)
        return ln_1001;

    auto low = approximation{1.0l, 0.0l};

    constexpr long double thousand_exp = 2.716923932235892457383l; // (not) coincidentally close to e
    while (low.value * thousand_exp < n) // speed up finding first value less than n
        low.value *= thousand_exp, low.exponent += 1000.0l;

    constexpr long double hundred_exp = 1.105115697720767968379l;
    while (low.value * hundred_exp < n) // speed up finding first value less than n
        low.value *= hundred_exp, low.exponent += 100.0l;

    while (low.value * 1.001l < n) // find the greatest power of 1.001 strictly less than n (1.001 ^ low.second < n)
        low.value *= 1.001l, low.exponent += 1.0l;

    auto high = low;
    while (high.value < n) // find the smallest power of 1.001 strictly greater than n (1.001 ^ high.second > n)
        high.value *= 1.001l, high.exponent += 1.0l;

    if (low.value == n) // if an exact value is found, return early
        return low.exponent * ln_1001;
    if (high.value == n) // if an exact value is found, return early
        return high.exponent * ln_1001;

    auto const t = (high.value - n) / (high.value - low.value); // for linear interpolation below
    auto const approx = (high.exponent - (high.exponent - low.exponent) * t); // approximately ln(n) / ln(1.001)
    return approx * ln_1001;
}

constexpr long double log(long double x) {
    if (x < 0) // if x < 0 ln(x) is undefined
        return std::numeric_limits<long double>::quiet_NaN();
    else if (x == 0) // if x == 0 ln(x) is -infinity
        return -std::numeric_limits<long double>::infinity();
    else if (x < 1.0) // can only find natural log of numbers greater than 1 and ln(x) = -ln(1 / x)
        return -ln_cp_impl(1.0l / x);
    else // normal case
        return ln_cp_impl(x);
}

constexpr long double log_n(long double x, long double n) {
    return log(x) / log(n);
}

constexpr long double log10(long double x) {
    return log_n(x, 10);
}

// tests

static_assert(lmj::abs(lmj::log(2) + lmj::log(3) - lmj::log(2 * 3)) < 1e-5);
}