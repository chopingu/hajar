#pragma once

#include "../../include.hpp"

namespace pinguml {

class cost_base {
public:
    virtual f32 cost([[maybe_unused]] const f32 output, [[maybe_unused]] const f32 target) = 0;
    virtual f32 cost_d([[maybe_unused]] const f32 output, [[maybe_unused]] const f32 target) = 0;
};

// ----- MSE ----- //

class mse : public cost_base {
public:
    virtual f32 cost([[maybe_unused]] const f32 output, [[maybe_unused]] const f32 target) {
        return 0.5f * (output - target) * (output - target);
    }

    virtual f32 cost_d([[maybe_unused]] const f32 output, [[maybe_unused]] const f32 target) {
        return output - target;
    }
};

// ----- CROSS ENTROPY ----- //

class cross_entropy : public cost_base {
    virtual f32 cost([[maybe_unused]] const f32 output, [[maybe_unused]] const f32 target) {
        return -target * std::log(output) - (1.f - target) * std::log(1.f - output);
    }

    virtual f32 cost_d([[maybe_unused]] const f32 output, [[maybe_unused]] const f32 target) {
        return (output - target) / (output * (1.f - output));
    }
};

cost_base *create_cost(const std::string type) {
    if(type == "mse") return new mse();
    else if (type == "cross_entropy") return new cross_entropy();
    else throw std::runtime_error("invalid cost-function '" + type + "'");

    return nullptr;
}

} // namespace pinguml
