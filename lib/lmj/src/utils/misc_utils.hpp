#pragma once

#include <string_view>
#include <thread>
#include "../containers/containers.hpp"

namespace lmj {
constexpr std::size_t compute_hash(char const *s, std::size_t size) { // FNV hash
    constexpr auto IS_64BIT = sizeof(std::size_t) == 8;
    std::size_t result = IS_64BIT ? 14695981039346656037ULL : 2166136261U;
    for (std::size_t i = 0; i < size; ++i) {
        result ^= s[i];
        result *= IS_64BIT ? 1099511628211ULL : 16777619U;
    }
    return result;
}

constexpr std::size_t seed_from_str(std::string_view v) {
    return compute_hash(&*v.begin(), v.size());
}

class constexpr_rand_generator { // based on xorshift random number generator by George Marsaglia
    unsigned long long x, y, z;

public:
    constexpr explicit constexpr_rand_generator(std::size_t seed = seed_from_str(__TIME__)) : x{}, y{}, z{} {
        set_seed(seed);
    }

    constexpr void set_seed(unsigned long long seed) {
        x = 230849599040350201 ^ static_cast<unsigned long long>(seed);
        y = 965937400815267857 ^ static_cast<unsigned long long>(seed);
        z = 895234450760720011 ^ static_cast<unsigned long long>(seed);
        for (int i = 0; i < 128; ++i)
            gen<int>(); // discard first 128 values
    }

    template<class T = std::size_t, typename = typename std::enable_if_t<std::is_integral_v<T>, void>>
    constexpr T gen() {
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        const std::size_t t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return static_cast<T>(z);
    }

    template<class T = std::size_t, typename = typename std::enable_if_t<std::is_integral_v<T>, void>>
    constexpr auto operator()() {
        return gen<T>();
    }
};

template<class T>
constexpr int sign(T const &x) {
    return (x > 0) - (x < 0);
}

namespace detail {
inline thread_local constexpr_rand_generator gen{seed_from_str(__TIME__) ^ [] {
    const auto id = std::this_thread::get_id();
    return *reinterpret_cast<std::uint64_t const *>(&id);
}()};
}

inline void srand(unsigned long long seed) {
    detail::gen.set_seed(seed);
}

template<class T = unsigned long long>
inline auto rand() -> std::enable_if_t<std::is_integral_v<T>, T> {
    return detail::gen.gen<T>();
}

constexpr auto random_shuffle(auto &random_access_container) {
    const auto n = random_access_container.size();
    using T = std::remove_cvref_t<decltype(n)>;
    for (T i = 0; i < n; ++i) {
        std::swap(random_access_container[i], random_access_container[rand<T>() % n]);
    }
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

template<class Iterable>
constexpr auto min(Iterable const &iterable) ->
std::remove_cvref_t<decltype(std::begin(iterable), std::end(iterable), std::declval<std::iter_value_t<Iterable>>())> {
    using T = std::iter_value_t<Iterable>;
    auto iter = std::begin(iterable), end = std::end(iterable);
    T result = *iter;
    while (++iter < end)
        result = min(result, *iter);
    return result;
}

template<class Iterable>
constexpr auto max(Iterable const &iterable) ->
std::remove_cvref_t<decltype(std::begin(iterable), std::end(iterable), std::declval<std::iter_value_t<Iterable>>())> {
    using T = std::iter_value_t<Iterable>;
    auto iter = std::begin(iterable), end = std::end(iterable);
    T result = *iter;
    while (++iter < end)
        result = max(result, *iter);
    return result;
}

// tests (very incomplete)

static_assert(min(1, 2) == 1);
static_assert(min(1, 2, 3) == 1);

static_assert(max(1, 2) == 2);
static_assert(max(1, 2, 3) == 3);

static_assert(sign(0) == 0);
static_assert(sign(1) == 1);
static_assert(sign(-1) == -1);

static_assert(min(std::array{1, 2, 3}) == 1);
static_assert(max(std::array{1, 2, 3}) == 3);
}
