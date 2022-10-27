#pragma once

#include <cmath>

#include "../defines.hpp"

// ----- MSE ----- //

auto mse = [](float output, float target) {
    return 0.5f * (output - target) * (output - target);
};

auto d_mse = [](float output, float target) {
    return output-target;
};

// ----- CROSS ENTROPY ----- //

auto cross_entropy = [](float output, float target) {
    return -target * std::log(output) - (1.f - target) * std::log(1.f - output);
};

auto d_cross_entropy = [](float output, float target) {
    return (output - target) / (output * (1.f - output));
};
