#pragma once

#include "../include.hpp"

template<class T, usize... sizes>
class weight_array {
    template<class G>
    struct matrix_ref {
        friend weight_array;
        G *data;
        usize subarray_len;
    private:
        constexpr matrix_ref(G *data, usize subarray_len) : data{data}, subarray_len{subarray_len} {}

    public:
        constexpr std::span<G> operator[](usize idx) {
            return std::span<G>{data + idx * subarray_len, data + (idx + 1) * subarray_len};
        }

        constexpr std::span<G const> operator[](usize idx) const {
            return std::span<G const>{data + idx * subarray_len, data + (idx + 1) * subarray_len};
        }
    };

private:
    constexpr static auto layer_sizes = std::array{sizes...};

    constexpr static auto indices = [] {
        std::array<usize, sizeof...(sizes)> arr{};
        usize sum = 0;
        for (usize i = 0; i < sizeof...(sizes) - 1; ++i) {
            arr[i] = sum;
            sum += layer_sizes[i] * layer_sizes[i + 1];
        }
        arr.back() = sum;
        return arr;
    }();

public:
    std::array<T, indices.back()> data{};

    constexpr matrix_ref<T> operator[](usize idx) {
        return matrix_ref<T>{data.data() + indices[idx], layer_sizes[idx + 1]};
    }

    constexpr matrix_ref<T const> operator[](usize idx) const {
        return matrix_ref<T const>{data.data() + indices[idx], layer_sizes[idx + 1]};
    }

    constexpr usize size() const {
        return sizeof...(sizes);
    }

    constexpr auto front() {
        return operator[](0);
    }

    constexpr auto front() const {
        return operator[](0);
    }

    constexpr auto back() {
        return operator[](size() - 1);
    }

    constexpr auto back() const {
        return operator[](size() - 1);
    }

    constexpr auto fill(T const &value) {
        data.fill(value);
    }
};
