#pragma once

#include <cmath>
#include <span>
#include <string>
#include <random>
#include <chrono>

#include "defines.hpp"

namespace util { // with external bias

std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());

template<class T>
void dirichlet_noise(std::span<T> input, std::span<const T> bias, const T alpha = 0.03f, const T eps = 0.25f) {
    std::gamma_distribution<T> gamma(alpha, 1);
    std::vector<T> noise;

    T sum = 0;
    for (u64 i = 0; i < input.size(); i++) {
        T x = gamma(gen);
        sum += x;
        noise.pb(x);
    }

    for (u64 i = 0; i < noise.size(); i++)
        noise[i] /= sum;

    sum = 0;
    for (u64 i = 0; i < input.size(); i++) {
        input[i] = (1.f - eps) * (input[i] + bias[i]) + eps * noise[i];
        sum += input[i];
    }

    for (u64 i = 0; i < input.size(); i++)
        input[i] /= sum;
}

template<class T>
void tanh(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        const T e1 = std::exp(x);
        const T e2 = std::exp(-x);
        input[i] = (e1 - e2) / (e1 + e2);
    }
}

template<class T>
T d_tanh(std::span<const T> input, const u64 index) {
    return (1.0f - input[index] * input[index]);
} // input[index] is already activated here

template<class T>
void identity(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++)
        input[i] = input[i] + bias[i];
}

template<class T>
T d_identity(std::span<const T> input, const u64 index) { return 1.0f; }

template<class T>
void elu(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        if (x < 0) input[i] = 0.1f * (std::exp(x) - 1.0f);
        else input[i] = x;
    }
}

template<class T>
T d_elu(std::span<const T> input, const u64 index) {
    if (input[index] > 0) return 1.0f;
    else return 0.1f * std::exp(input[index]);
}

template<class T>
void relu(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        if (x < 0) input[i] = 0;
        else input[i] = x;
    }
}

template<class T>
T d_relu(std::span<const T> input, const u64 index) {
    if (input[index] > 0) return 1.0f;
    else return 0.0f;
}

template<class T>
void lrelu(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        if (x < 0) input[i] = 0.01f * x;
        else input[i] = x;
    }
}

template<class T>
T d_lrelu(std::span<const T> input, const u64 index) {
    if (input[index] > 0) return 1.0f;
    else return 0.01f;
}

template<class T>
void vlrelu(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        if (x < 0) input[i] = 0.2f * x;
        else input[i] = x;
    }
}

template<class T>
T d_vlrelu(std::span<const T> input, const u64 index) {
    if (input[index] > 0) return 1.0f;
    else return 0.2f;
}

template<class T>
void sigmoid(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        input[i] = 1.0f / (1.0f + std::exp(-x));
    }
}

template<class T>
T d_sigmoid(std::span<const T> input, const u64 index) {
    return input[index] * (1.0f - input[index]);
} // input[index] is already activated

template<class T>
void softmax(std::span<T> input, std::span<const T> bias, const u64 _size) {
    T mx = input[0] + bias[0];
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        if (mx < x) mx = x;
    }
    T sum = 0;
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        sum += std::exp(x - mx);
    }

    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        input[i] = std::exp(x - mx) / sum;
    }
}

template<class T>
T d_softmax(std::span<const T> input, const u64 index) {
    return input[index] * (1.0f - input[index]);
} // input[index] is already activated

} // namespace util

namespace util { // without external bias


template<class T>
void dirichlet_noise(std::span<T> input, const T alpha, const T eps) {
    std::gamma_distribution<T> gamma(alpha, 1);
    std::vector<T> noise;

    T sum = 0;
    for (u64 i = 0; i < input.size(); i++) {
        T x = gamma(gen);
        sum += x;
        noise.pb(x);
    }

    for (u64 i = 0; i < noise.size(); i++)
        noise[i] /= sum;

    sum = 0;
    for (u64 i = 0; i < input.size(); i++) {
        input[i] = (1.f - eps) * input[i] + eps * noise[i];
        sum += input[i];
    }

    for (u64 i = 0; i < input.size(); i++)
        input[i] /= sum;
}

template<class T>
void tanh(std::span<T> input, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        const T e1 = std::exp(x);
        const T e2 = std::exp(-x);
        input[i] = (e1 - e2) / (e1 + e2);
    }
}

template<class T>
void identity(std::span<T> input, const u64 _size) {
    return;
}

template<class T>
void elu(std::span<T> input, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        if (x < 0) input[i] = 0.1f * (std::exp(x) - 1.0f);
        else input[i] = x;
    }
}

template<class T>
void relu(std::span<T> input, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        if (x < 0) input[i] = 0;
        else input[i] = x;
    }
}

template<class T>
void lrelu(std::span<T> input, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        if (x < 0) input[i] = 0.01f * x;
        else input[i] = x;
    }
}

template<class T>
void vlrelu(std::span<T> input, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        if (x < 0) input[i] = 0.2f * x;
        else input[i] = x;
    }
}

template<class T>
void sigmoid(std::span<T> input, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        input[i] = 1.0f / (1.0f + std::exp(-x));
    }
}

template<class T>
void softmax(std::span<T> input, const u64 _size) {
    T mx = 0;
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        if (mx < x) mx = x;
    }

    T sum = 0;
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        sum += std::exp(x - mx);
    }

    for (u64 i = 0; i < _size; i++) {
        const T x = input[i];
        input[i] = std::exp(x - mx) / sum;
    }
}
} // namespace util
