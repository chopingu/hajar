#pragma once

#include <cmath>
#include <span>
#include <string>
#include <array>
#include "../defines.hpp"

namespace pingu {

namespace tan_h {
std::string name = "tan_h";

template<class T>
void activation(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        const T x = input[i] + bias[i];
        const T e1 = std::exp(x);
        const T e2 = std::exp(-x);
        input[i] = (e1 - e2) / (e1 + e2);
    }
}

template<class T>
T d_activation(std::span<const T> input, const u64 index) {
    return (1.0f - input[index] * input[index]);
} // input is already activated here
}

namespace identity {
constexpr auto name = "identity";

template<class T>
void activation(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++)
        input[i] = input[i] + bias[i];
}

template<class T>
T d_activation(std::span<const T> input, const u64 ind) { return 1.0f; }
}

namespace elu {
constexpr auto name = "elu";

template<class T>
void activation(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        T x = input[i] + bias[i];
        if (x < 0) input[i] = 0.1f * (std::exp(x) - 1.0f);
        else input[i] = x;
    }
}

template<class T>
T d_activation(std::span<const T> input, const u64 i) {
    if (input[i] > 0) return 1.0f;
    else return 0.1f * std::exp(input[i]);
}
}

namespace relu {
constexpr auto name = "relu";

template<class T>
void activation(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        T x = input[i] + bias[i];
        if (x < 0) input[i] = 0;
        else input[i] = x;
    }
}

template<class T>
T d_activation(std::span<const T> input, const u64 i) {
    if (input[i] > 0) return 1.0f;
    else return 0.0f;
}
}

namespace lrelu {
constexpr auto name = "lrelu";

template<class T>
void activation(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        T x = input[i] + bias[i];
        if (x < 0) input[i] = 0.01f * x;
        else input[i] = x;
    }
}

template<class T>
T d_activation(std::span<const T> input, const u64 i) {
    if (input[i] > 0) return 1.0f;
    else return 0.01f;
}
}

namespace vlrelu {
constexpr auto name = "vlrelu";

template<class T>
void activation(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        T x = input[i] + bias[i];
        if (x < 0) input[i] = 0.2f * x;
        else input[i] = x;
    }
}

template<class T>
T d_activation(std::span<const T> input, const u64 i) {
    if (input[i] > 0) return 1.0f;
    else return 0.2f;
}
}

namespace sigmoid {
constexpr auto name = "sigmoid";

template<class T>
void activation(std::span<T> input, std::span<const T> bias, const u64 _size) {
    for (u64 i = 0; i < _size; i++) {
        T x = input[i] + bias[i];
        input[i] = 1.0f / (1.0f + std::exp(-x));
    }
}

template<class T>
T d_activation(std::span<const T> input, const u64 i) {
    return input[i] * (1.0f - input[i]);
} // input[i] is already activated
}

namespace none {
constexpr auto name = "none";

template<class T>
void activation(std::span<T> input, std::span<const T> bias, const u64 _size) { return; }

template<class T>
T d_activation(std::span<const T> input, const u64 ind) { return; }
}

template<class T>
struct activation_function {
    const char *name;

    void (*activation)(std::span<T>, std::span<const T>, const u64);

    T (*d_activation)(std::span<const T>, const u64);
};

template<class T>
activation_function<T> *new_activation_function(std::string_view new_function) {
    activation_function<T> *nw = new activation_function<T>;

    if (new_function == tan_h::name) {
        nw->activation = &tan_h::activation;
        nw->d_activation = &tan_h::d_activation;
        nw->name = tan_h::name;
        return nw;
    }
    if (new_function == identity::name) {
        nw->activation = &identity::activation;
        nw->d_activation = &identity::d_activation;
        nw->name = identity::name;
        return nw;
    }
    if (new_function == elu::name) {
        nw->activation = &elu::activation;
        nw->d_activation = &elu::d_activation;
        nw->name = elu::name;
        return nw;
    }
    if (new_function == relu::name) {
        nw->activation = &relu::activation;
        nw->d_activation = &relu::d_activation;
        nw->name = relu::name;
        return nw;
    }
    if (new_function == lrelu::name) {
        nw->activation = &lrelu::activation;
        nw->d_activation = &relu::d_activation;
        nw->name = relu::name;
        return nw;
    }
    if (new_function == vlrelu::name) {
        nw->activation = &vlrelu::activation;
        nw->d_activation = &vlrelu::d_activation;
        nw->name = vlrelu::name;
        return nw;
    }
    if (new_function == sigmoid::name) {
        nw->activation = &sigmoid::activation;
        nw->d_activation = &sigmoid::d_activation;
        nw->name = sigmoid::name;
        return nw;
    }

    delete nw;
    return 0;
}
}
