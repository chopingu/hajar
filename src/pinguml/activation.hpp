#pragma once

#include "../include.hpp"

// ADD SIMD SUPPORT

namespace pinguml {

// ----- null ----- //

auto null = [](f32 *input, const f32 *biases, const u32 size) {
    return;
};

auto null_c = [](f32 *input, const f32 bias, const u32 size) {
    return;
};

// ----- identity ----- //

auto identity = [](f32 *input, const f32 *biases, const u32 size) {
    for(u32 i = 0; i < size; i++) 
        input[i] = input[i] + biases[i];
};

auto identity_c = [](f32 *input, const f32 bias, const u32 size) {
    for(u32 i = 0; i < size; i++) 
        input[i] = input[i] + bias;
};

// ----- tanh ------ //

auto tanh = [](f32 *input, const f32 *biases, const u32 size) {
    for(u32 i = 0; i < size; i++) 
        input[i] = std::tanh(input[i] + biases[i]);
};

auto tanh_c = [](f32 *input, const f32 bias, const u32 size) {
    for(u32 i = 0; i < size; i++) 
        input[i] = std::tanh(input[i] + bias);
};

// ----- elu ----- //

auto elu = [](f32 *input, const f32 *biases, const u32 size) {
    for(u32 i = 0; i < size; i++) {
        const f32 x = input[i] + biases[i];
        if(x < 0.f) input[i] = 0.1f * (std::exp(x) - 1.f);
        else input[i] = x;
    }
};

auto elu_c = [](f32 *input, const f32 bias, const u32 size) {
    for(u32 i = 0; i < size; i++) {
        const f32 x = input[i] + bias;
        if(x < 0.f) input[i] = 0.1f * (std::exp(x) - 1.f);
        else input[i] = x;
    }
};

// ----- relu ----- //

auto relu = [](f32 *input, const f32 *biases, const u32 size) {
    for(u32 i = 0; i < size; i++) {
        const f32 x = input[i] + biases[i];
        if(x < 0.f) input[i] = 0.f;
        else input[i] = x;
    }
};

auto relu_c = [](f32 *input, const f32 bias, const u32 size) {
    for(u32 i = 0; i < size; i++) {
        const f32 x = input[i] + bias;
        if(x < 0.f) input[i] = 0.f;
        else input[i] = x;
    }
};

// ----- lrelu ----- //

auto lrelu = [](f32 *input, const f32 *biases, const u32 size) {
    for(u32 i = 0; i < size; i++) {
        const f32 x = input[i] + biases[i];
        if(x < 0.f) input[i] = 0.01f * x;
        else input[i] = x;
    }
};

auto lrelu_c = [](f32 *input, const f32 bias, const u32 size) {
    for(u32 i = 0; i < size; i++) {
        const f32 x = input[i] + bias;
        if(x < 0.f) input[i] = 0.01f * x;
        else input[i] = x;
    }
};

// ----- vlrelu ----- //

auto vlrelu = [](f32 *input, const f32 *biases, const u32 size) {
    for(u32 i = 0; i < size; i++) {
        const f32 x = input[i] + biases[i];
        if(x < 0.f) input[i] = 0.3f * x;
        else input[i] = x;
    }
};

auto vlrelu_c = [](f32 *input, const f32 bias, const u32 size) {
    for(u32 i = 0; i < size; i++) {
        const f32 x = input[i] + bias;
        if(x < 0.f) input[i] = 0.3f * x;
        else input[i] = x;
    }
};

// ----- sigmoid ----- //

auto sigmoid = [](f32 *input, const f32 *biases, const u32 size) {
    for(u32 i = 0; i < size; i++) 
        input[i] = 1.0f / (1.0f + std::exp(-(input[i] + biases[i])));
};

auto sigmoid_c = [](f32 *input, const f32 bias, const u32 size) {
    for(u32 i = 0; i < size; i++) 
        input[i] = 1.0f / (1.0f + std::exp(-(input[i] + bias)));
};

} // namespace pinguml