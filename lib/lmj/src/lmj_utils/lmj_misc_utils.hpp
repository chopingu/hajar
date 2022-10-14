#pragma once

#include <string_view>
#include "../lmj_containers/lmj_containers.hpp"

namespace lmj {
constexpr unsigned long long seed_from_str(std::string_view v) {
    unsigned long long ans = 0;
    for (auto &&c: v)
        ans = (ans << 32 | ans >> 32) ^ c;
    return ans;
}

template<class T = unsigned long long, typename = typename std::enable_if_t<std::is_integral_v<T>, void>>
class constexpr_rand_generator { // based on xorshift random number generator by George Marsaglia
    unsigned long long x, y, z;

public:
    constexpr explicit constexpr_rand_generator(unsigned long long seed = seed_from_str(__TIME__)) : x{}, y{}, z{} {
        set_seed(seed);
    }

    constexpr auto set_seed(unsigned long long seed) {
        x = 230849599040350201 ^ static_cast<unsigned long long>(seed);
        y = 965937400815267857 ^ static_cast<unsigned long long>(seed);
        z = 895234450760720011 ^ static_cast<unsigned long long>(seed);
        for (int i = 0; i < 128; ++i)
            gen(); // discard first 128 values
    }

    constexpr auto gen() {
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        auto t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return static_cast<T>(z);
    }

    constexpr auto operator()() {
        return gen();
    }
};

template<class T>
constexpr int sign(T const &x) {
    return (x > 0) - (x < 0);
}

namespace _gen_namespace {
template<class T>
constexpr_rand_generator<T> gen{seed_from_str(__TIME__)};
}

template<class T = unsigned long long>
void srand(unsigned long long seed) {
    _gen_namespace::gen<T>.set_seed(seed);
}

template<class T = unsigned long long>
auto rand() -> std::enable_if_t<std::is_integral_v<T>, T> {
    return _gen_namespace::gen<T>();
}

template<class T, class G>
constexpr auto min(T const &a, G const &b) noexcept -> decltype(a + b) {
    return a < b ? a : b;
}

template<class T, class G>
constexpr auto max(T const &a, G const &b) noexcept -> decltype(a + b) {
    return a > b ? a : b;
}

template<class T, class... G>
constexpr auto min(T const &a, G const &... b) noexcept {
    return min(a, min(b...));
}

template<class T, class... G>
constexpr auto max(T const &a, G const &... b) noexcept {
    return max(a, max(b...));
}

// tests (very incomplete)

static_assert(min(1, 2) == 1);
static_assert(min(1, 2, 3) == 1);

static_assert(max(1, 2) == 2);
static_assert(max(1, 2, 3) == 3);

static_assert(sign(0) == 0);
static_assert(sign(1) == 1);
static_assert(sign(-1) == -1);
}