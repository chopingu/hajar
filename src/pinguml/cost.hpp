#pragma once

#include "../include.hpp"

namespace pinguml {

// ----- MSE ----- //

auto mse = [](const f32 output, const f32 target) {
    return 0.5f * (output - target) * (output - target);
};

auto d_mse = [](const f32 output, const f32 target) {
    return output - target;
};

// ----- CROSS ENTROPY ----- //

auto cross_entropy = [](const f32 output, const f32 target) {
    return -target * std::log(output) - (1.f - target) * std::log(1.f - output);
};

auto d_cross_entropy = [](const f32 output, const f32 target) {
    return (output - target) / (output * (1.f - output));
};

} // namespace pinguml
