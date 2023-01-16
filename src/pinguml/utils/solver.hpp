#pragma once

#include "../../include.hpp"

namespace pinguml {

class solver_base {
public:
    f32 m_learning_rate;

    solver_base() : m_learning_rate(0.1f) {}

    virtual ~solver_base() {}

    virtual void reset() {}

    virtual void push_back([[maybe_unused]] const u32 h, [[maybe_unused]] const u32 w, [[maybe_unused]] const u32 c) {}

    virtual void update_weights([[maybe_unused]] tensor *weights, [[maybe_unused]] const u32 index, [[maybe_unused]] const tensor &delta_weights, [[maybe_unused]] const f32 alpha = 1.0f) {}
};

class sgd : public solver_base {
public:
    virtual void update_weights(tensor *weights, [[maybe_unused]] const u32 index, const tensor &delta_weights, [[maybe_unused]] const f32 alpha = 1.0f) {
        const f32 weight_decay = 0.01f;
        const f32 learning_rate = weight_decay * m_learning_rate;
        for(u32 i = 0; i < weights->size(); i++)	
            weights->m_ptr[i] -= learning_rate * (delta_weights.m_ptr[i] + weight_decay * weights->m_ptr[i]);
    }
};

solver_base* create_solver(const std::string name) {
    if(name == "null") return nullptr;
    else if(name == "sgd") { return new sgd();}

    throw std::runtime_error("invalid name '" + name + "'");
}

} // namespace pinguml
