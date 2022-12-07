#pragma once

#include "../include.hpp"

template<class T, usize... sizes>
class layer_array {
private:
    constexpr static auto layer_sizes = std::array{sizes...};

    constexpr static auto indices = [] {
        std::array<usize, sizeof...(sizes)> arr{};
        for (usize i = 0, sum = 0; i < arr.size(); ++i) {
            arr[i] = sum;
            sum += std::array{sizes...}[i];
        }
        return arr;
    }();
public:
    std::array<T, (sizes + ...)> m_data{};

    constexpr std::span<T> operator[](usize idx) {
        return std::span<T>(m_data.data() + indices[idx], m_data.data() + indices[idx] + layer_sizes[idx]);
    }

    constexpr std::span<T const> operator[](usize idx) const {
        return std::span<T const>(m_data.data() + indices[idx], m_data.data() + indices[idx] + layer_sizes[idx]);
    }

    constexpr usize size() const {
        return sizeof...(sizes);
    }

    constexpr std::span<T> front() {
        return operator[](0);
    }

    constexpr std::span<T const> front() const {
        return operator[](0);
    }

    constexpr std::span<T> back() {
        return operator[](size() - 1);
    }

    constexpr std::span<T const> back() const {
        return operator[](size() - 1);
    }

    constexpr auto fill(T const &value) {
        m_data.fill(value);
    }

    constexpr auto data() {
        return m_data.data();
    }

    constexpr auto data() const {
        return m_data.data();
    }
};
