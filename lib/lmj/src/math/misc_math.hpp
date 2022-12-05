#pragma once

#include "../utils/concepts.hpp"
#include "newton_raphson.hpp"
#include "../containers/container_helpers.hpp"

#include <cmath>
#include <ranges>
#include <cassert>

namespace lmj {

#if defined(__GNUC__) || defined(__clang_major__) || defined (__clang_minor__)
using biggest_float = __float128;
#else
using biggest_float = long double;
#endif

template<class T = long double>
constexpr T e = 2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427427l;

template<class T = long double>
constexpr T pi = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798l;

/**
 * @param base
 * @param exp
 * @throws std::out_of_range if exp = base = 0
 * @return base ^ exp
 */
template<class T>
requires number<T>
constexpr auto ipow(T base, std::uint64_t exp) {
    if (exp == 1)
        return base;
    if (exp == 0 && base != 0)
        return T{1};
    if (exp == 0 && base == 0)
        throw std::out_of_range("0^0 is undefined");
    T result = 1;
    while (exp) {
        result *= (exp & 1) ? base : T{1};
        exp >>= 1;
        base *= base;
    }
    return result;
}

/**
 * @param x
 * @return |x|
 */
template<class T>
requires number<T>
constexpr T abs(T x) {
    if ((std::is_constant_evaluated() && !std::is_floating_point_v<T>) ||
        (std::is_same_v<T, long double> && sizeof(long double) == 16)) {
        return x < 0 ? -x : x;
    } else {
        if constexpr (std::endian::native == std::endian::little) {
            // clear sign bit
            if (!std::is_constant_evaluated())
                ((std::uint8_t *) &x)[sizeof(T) - 1] &= ~0x80;
        } else if constexpr (std::endian::native == std::endian::big) {
            // clear sign bit
            if (!std::is_constant_evaluated())
                ((std::uint8_t *) &x)[0] &= ~0x80;
        } else {
            // wtf is your architecture
            static_assert(std::endian::native == std::endian::big ||
                          std::endian::native == std::endian::little);
        }
        return x;
    }
}

constexpr biggest_float _exp_small(biggest_float x, int n = 32) {
    biggest_float sum = 1;
    while (--n)
        sum = 1 + x * sum / static_cast<biggest_float>(n);
    return sum;
}

/**
 * @param x
 * @return e ^ x
 */
constexpr long double exp(long double x) {
    if (std::is_constant_evaluated()) {
        assert(-11356 <= x && x <= 11356);
        if (x == 0)
            return 1;
        if (x < 0)
            return 1.0l / lmj::exp(-x);
        const auto whole_part = static_cast<unsigned>(x);
        const auto fractional_part = x - whole_part;
        return static_cast<long double>(
                _exp_small(fractional_part) * ipow(e<biggest_float>, whole_part)
        );
    } else {
        return std::exp(x);
    }
}

/**
 * @param x
 * @return floor(log2(x))
 */
constexpr auto flog2(integral auto x) {
    if (x <= 0)
        return std::numeric_limits<int>::min();
    int ans = 0;
    while (x >>= 1)
        ++ans;
    return ans;
}

/**
 * @param x
 * @return floor(log2(x))
 */
constexpr auto flog2(floating_point auto x) {
    using T = std::remove_cvref_t<decltype(x)>;
    if (x < 0)
        return std::numeric_limits<T>::quiet_NaN();
    if (x == 0)
        return -std::numeric_limits<T>::infinity();
    std::int64_t ans = 0;
    while ((x /= 2) >= 1)
        ++ans;
    return static_cast<T>(ans);
}

/**
 * @param x
 * @return floor(sqrt(x))
 */
constexpr auto integral_sqrt(const integral auto x) {
    using T = std::remove_cvref_t<decltype(x)>;
    if (x == 0) return T{0};
    const auto l = flog2(x);
    T low = 1ull << (l / 2), high = static_cast<T>(1) << (l / 2 + 1);
    while (low + 1 < high) {
        auto mid = high - (high - low) / 2;
        if (mid <= x / mid)
            low = mid;
        else
            high = mid;
    }
    return low;
}

/**
 * @param x
 * @return floor(sqrt(x))
 */
constexpr auto integral_sqrt(const floating_point auto x) {
    using T = std::uint64_t;
    if (static_cast<T>(x) == 0) return T{0};
    const auto l = flog2(static_cast<T>(x));
    T low = 1ull << (l / 2), high = 1ull << (l / 2 + 1);
    while (low + 1 < high) {
        auto mid = high - (high - low) / 2;
        if (mid <= x / mid)
            low = mid;
        else
            high = mid;
    }
    return low;
}

/**
 * @param x
 * @throws std::out_of_range if x < 0
 * @return square root of x
 */
constexpr auto sqrt(number auto x) {
    if (x < 0)
        throw std::out_of_range("can't take square root of negative number");
    if (std::is_constant_evaluated()) {
        long double root = integral_sqrt(x);
        long double dx;
        do {
            dx = (root * root - x) / (2 * root);
            root -= dx;
        } while (dx > 1e-5 || -dx > 1e-5);
        auto extra_iterations = 4;
        while (extra_iterations--)
            root -= (root * root - x) / (2 * root);
        return root;
    } else {
        return std::sqrt(static_cast<long double>(x));
    }
}

constexpr auto hypot(numbers auto... nums) {
    const auto sum = ((nums * nums) + ... + 0);
    if (std::is_constant_evaluated()) {
        return lmj::sqrt(static_cast<long double>(sum));
    } else {
        return std::sqrt(static_cast<long double>(sum));
    }
}

/**
 * @param f integrand
 * @param low lower limit of integration
 * @param high upper limit of integration
 * @param steps number of steps between limits
 * @return integral of f from low to high
 */
constexpr auto integrate(auto &&f, long double low, long double high, std::uint64_t steps = 1e6) {
    long double sum = 0;
    long double last_y = f(low);
    const long double step_size = (high - low) / static_cast<long double>(steps);
    for (std::uint64_t step = 1; step < steps; ++step) {
        const long double y = f(low + step_size * step);
        sum += y + last_y;
        last_y = y;
    }
    return sum * step_size / 2.0l;
}

constexpr auto sigma(std::uint64_t n) {
    std::uint64_t res = 1;
    for (std::uint64_t p = 2; p * p <= n; p += 1 + (p & 1)) {
        std::uint64_t a = 0;
        while (n % p == 0)
            n /= p, ++a;
        if (a)
            res *= (ipow(p, a + 1) - 1) / (p - 1);
    }
    if (n > 1)
        res *= n + 1;
    return res;
}

constexpr auto euler_totient(std::uint64_t n) {
    std::uint64_t res = n;
    for (std::uint64_t p = 2; p * p <= n; p += 1 + (p & 1)) {
        std::uint64_t a = 0;
        while (n % p == 0)
            n /= p, ++a;
        if (a)
            res /= p, res *= p - 1;
    }
    if (n > 1)
        res /= n, res *= n - 1;
    return res;
}

constexpr std::pair<std::uint64_t, std::uint64_t> farey(long double x, std::uint64_t limit) {
    if (x > 1) {
        const auto whole_part = static_cast<std::uint64_t>(x);
        auto [num, denom] = farey(x - whole_part, limit);
        num += denom * whole_part;
        return {num, denom};
    }
    std::uint64_t a = 0, b = 1, c = 1, d = 1, ac = 1, bd = 2;
    while (bd <= limit) {
        if (x * bd > ac) {
            a = ac;
            b = bd;
        } else {
            c = ac;
            d = bd;
        }
        if (x - static_cast<long double>(a) / b == 0)
            return {a, b};
        if (x - static_cast<long double>(c) / d == 0)
            return {c, d};
        ac = a + c;
        bd = b + d;
    }
    long double ab = abs(x - static_cast<long double>(a) / b),
            cd = abs(x - static_cast<long double>(c) / d);
    if (ab > cd) {
        return {c, d};
    } else {
        return {a, b};
    }

}

// tests

static_assert(lmj::abs(lmj::integrate([](auto x) { return x * x; }, 0, 3, 1e5) - 9) < 1e-3);
static_assert(lmj::hypot(3, 4) == 5);
static_assert(lmj::sqrt(25) == 5);
static_assert(lmj::sqrt(9) == 3);
static_assert(lmj::integral_sqrt(2147483647) == 46340);
static_assert(lmj::ipow(0.5, 4) == 0.0625);
static_assert(lmj::abs(lmj::exp(10) - 22026.465794806716516) < 1e-5);
static_assert([] {
    for (int i = 0; i < 10000; ++i) {
        int root = lmj::integral_sqrt(i);
        if (!(root * root <= i && (root + 1) * (root + 1) > i))
            return false;
    }
    return true;
}());
static_assert([] {
    for (int i = 1; i < 10000; ++i) {
        int l = lmj::flog2(i);
        if (!((1 << l) <= i && (1 << (l + 1)) > i))
            return false;
    }
    return true;
}());
}