#pragma once

#include <type_traits>
#include <functional>

namespace lmj {
template<class T> concept integral = std::is_integral_v<std::remove_cvref_t<T>>;
template<class T> concept unsigned_integral = integral<T> &&
                                              !std::is_signed_v<std::remove_cvref_t<T>>;
template<class T> concept signed_integral = integral<T> &&
                                            std::is_signed_v<std::remove_cvref_t<T>>;

template<class T> concept floating_point = std::is_floating_point_v<std::remove_cvref_t<T>>;

template<class T>
concept number = std::is_integral_v<std::remove_cvref_t<T>> ||
                 std::is_floating_point_v<std::remove_cvref_t<T>>;

template<class T> concept iterable = requires(T x) {
    std::begin(x);
    std::end(x);
};

template<class T> concept not_iterable = !iterable<T>;

template<class T, class... G>
constexpr bool number_helper() {
    if constexpr (sizeof...(G)) {
        return number<T> && number_helper<G...>();
    } else {
        return number<T>;
    }
}

template<class... T> concept numbers = number_helper<T...>();

static_assert(number<int>);
static_assert(number<float>);
static_assert(number<const int &>);
static_assert(number<const float &>);
static_assert(integral<int>);
static_assert(signed_integral<signed>);
static_assert(unsigned_integral<unsigned>);
static_assert(!signed_integral<unsigned>);
static_assert(!unsigned_integral<signed>);
static_assert(numbers<char, int, float, double>);
}