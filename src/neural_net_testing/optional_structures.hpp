#pragma once

#include "layer_array.hpp"
#include "weight_array.hpp"
#include "../defines.hpp"
#include <type_traits>

namespace gya {
template<bool used, class T, u64... sizes>
struct optional_layer_array {
private:
    struct empty_struct {
    };

public:
    using layer_array_t = std::conditional_t<used, layer_array<T, sizes...>, empty_struct>;
    layer_array_t m_layer_array;

    constexpr std::span<T> operator[](u64 idx) {
        static_assert(used);
        return m_layer_array.operator[](idx);
    }

    constexpr std::span<T const> operator[](u64 idx) const {
        static_assert(used);
        return m_layer_array.operator[](idx);
    }

    constexpr std::size_t size() const {
        static_assert(used);
        return m_layer_array.size();
    }

    constexpr std::span<T> front() {
        static_assert(used);
        return m_layer_array.front();
    }

    constexpr std::span<T const> front() const {
        static_assert(used);
        return m_layer_array.front();
    }

    constexpr std::span<T> back() {
        static_assert(used);
        return m_layer_array.back();
    }

    constexpr std::span<T const> back() const {
        static_assert(used);
        return m_layer_array.back();
    }

    constexpr bool active() const {
        return used;
    }

    constexpr auto fill(T const &value) {
        static_assert(used);
        m_layer_array.fill(value);
    }
};

template<bool used, class T, u64... sizes>
struct optional_weight_array {
private:
    struct empty_struct {
    };

public:
    using weight_array_t = std::conditional_t<used, weight_array<T, sizes...>, empty_struct>;
    weight_array_t m_weight_array;

    constexpr auto operator[](u64 idx) {
        static_assert(used);
        return m_weight_array.operator[](idx);
    }

    constexpr auto operator[](u64 idx) const {
        static_assert(used);
        return m_weight_array.operator[](idx);
    }

    constexpr std::size_t size() const {
        static_assert(used);
        return m_weight_array.size();
    }

    constexpr auto front() {
        static_assert(used);
        return m_weight_array.front();
    }

    constexpr auto front() const {
        static_assert(used);
        return m_weight_array.front();
    }

    constexpr auto back() {
        static_assert(used);
        return m_weight_array.back();
    }

    constexpr auto back() const {
        static_assert(used);
        return m_weight_array.back();
    }

    constexpr bool active() const {
        return used;
    }

    constexpr auto fill(T const &value) {
        static_assert(used);
        m_weight_array.fill(value);
    }
};
}