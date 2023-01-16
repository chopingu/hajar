#pragma once

#include "../../include.hpp"

namespace pinguml {

class cost_base {
public:
    std::string m_name;

    virtual ~cost_base() {}

    virtual f32 cost([[maybe_unused]] const f32 output, [[maybe_unused]] const f32 target) = 0;
    virtual f32 cost_d([[maybe_unused]] const f32 output, [[maybe_unused]] const f32 target) = 0;
};

// ----- MSE ----- //

class mse : public cost_base {
public:
    mse() : cost_base() {
        m_name = "mse";
    }

    virtual ~mse() final override {}

    virtual f32 cost(const f32 output, const f32 target) final override {
        return 0.5f * (output - target) * (output - target);
    }

    virtual f32 cost_d(const f32 output, const f32 target) final override {
        return output - target;
    }
};

// ----- CROSS ENTROPY ----- //

class cross_entropy : public cost_base {
public:
    cross_entropy() : cost_base() {
        m_name = "cross_entropy";
    }

    virtual ~cross_entropy() final override {}

    virtual f32 cost(const f32 output, const f32 target) final override {
        return -target * std::log(output) - (1.f - target) * std::log(1.f - output);
    }

    virtual f32 cost_d(const f32 output, const f32 target) final override {
        return (output - target) / (output * (1.f - output));
    }
};

cost_base *create_cost(std::string_view type) {
    if (type == "mse")
        return new mse();
    else if (type == "cross_entropy")
        return new cross_entropy();
    else
        throw std::runtime_error("invalid cost-function '" + std::string{type} + "'");
}

} // namespace pinguml
