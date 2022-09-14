#pragma once

#include "defines.hpp"
#include <array>
#include <span>

template<class T, u64... sizes>
class weight_array {
    template<class G>
    struct matrix_ref {
        friend weight_array;
        G *data;
        u64 subarray_len;
    private:
        constexpr matrix_ref(G *data, u64 subarray_len) : data{data}, subarray_len{subarray_len} {}

    public:
        constexpr std::span<G> operator[](u64 idx) {
            return std::span<G>{data + idx * subarray_len, data + (idx + 1) * subarray_len};
        }

        constexpr std::span<G const> operator[](u64 idx) const {
            return std::span<G const>{data + idx * subarray_len, data + (idx + 1) * subarray_len};
        }
    };

private:
    constexpr static auto layer_sizes = std::array{sizes...};

    constexpr static auto indices = [] {
        std::array<u64, sizeof...(sizes)> arr{};
        u64 sum = 0;
        for (u64 i = 0; i < sizeof...(sizes) - 1; ++i) {
            arr[i] = sum;
            sum += layer_sizes[i] * layer_sizes[i + 1];
        }
        arr.back() = sum;
        return arr;
    }();

public:
    std::array<T, indices.back()> data{};

    constexpr matrix_ref<T> operator[](u64 idx) {
        return matrix_ref<T>{data.data() + indices[idx], layer_sizes[idx + 1]};
    }

    constexpr matrix_ref<T const> const operator[](u64 idx) const {
        return matrix_ref<T const>{data.data() + indices[idx], layer_sizes[idx + 1]};
    }
};
