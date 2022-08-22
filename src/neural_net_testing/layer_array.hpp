#pragma once

#include "defines.hpp"
#include <array>
#include <span>

template<class T, u64... sizes>
class layer_array {
private:
    constexpr static auto layer_sizes = std::array{sizes...};

    constexpr static auto indices = [] {
        std::array<u64, sizeof...(sizes)> arr{};
        for (u64 i = 0, sum = 0; i < arr.size(); ++i) {
            arr[i] = sum;
            sum += std::array{sizes...}[i];
        }
        return arr;
    }();
public:
    std::array<T, (sizes + ...)> data;

    constexpr std::span<T> operator[](u64 idx) {
        return std::span<T>(data.data() + indices[idx], data.data() + indices[idx] + layer_sizes[idx]);
    }

    constexpr std::span<T const> operator[](u64 idx) const {
        return std::span<T const>(data.data() + indices[idx], data.data() + indices[idx] + layer_sizes[idx]);
    }

    constexpr std::size_t size() const {
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
};
