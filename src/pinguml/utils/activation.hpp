#pragma once

#include "../../include.hpp"

namespace pinguml {

// ----- activation_base ----- //

class activation_base {
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) = 0;
    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) = 0;
    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) = 0;
};

// ----- null ----- //

class null : public activation_base {
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) {
        return;
    }

    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) {
        return;
    }

    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) {
        return 0.f;
    }
};

// ----- identity ----- //

class identity : public activation_base {
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) 
            input[i] += biases[i];
    }

    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) 
            input[i] += bias;
    }

    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) {
        return 1.f;
    }
};

// ----- tanh ------ //

class tanh : public activation_base {
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) 
            input[i] = std::tanh(input[i] + biases[i]);
    }

    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) 
            input[i] = std::tanh(input[i] + bias);
    }

    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) {
        return 1.f - x[index] * x[index];
    }
};

// ----- elu ----- //

class elu : public activation_base {
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) {
            const f32 x = input[i] + biases[i];
            if(x < 0.f) input[i] = 0.1f * (std::exp(x) - 1.f);
            else input[i] = x;
        }
    }

    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) {
            const f32 x = input[i] + bias;
            if(x < 0.f) input[i] = 0.1f * (std::exp(x) - 1.f);
            else input[i] = x;
        }
    }
    
    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) { // post-bias pre-activation?
        if(x[index] >= 0) return 1.f;
        return 0.1f * std::exp(x[index]);
    }
};

// ----- relu ----- //

class relu : public activation_base {
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) {
            const f32 x = input[i] + biases[i];
            if(x < 0.f) input[i] = 0.f;
            else input[i] = x;
        }
    }

    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) {
            const f32 x = input[i] + bias;
            if(x < 0.f) input[i] = 0.f;
            else input[i] = x;
        }
    }

    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) { // post-bias pre-activation?
        if(x[index] >= 0) return 1.f;
        return 0.f;
    }
};

// ----- lrelu ----- //

class lrelu : public activation_base {
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) {
            const f32 x = input[i] + biases[i];
            if (x < 0.f) input[i] = 0.01f * x;
            else input[i] = x;
        }
    }

    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) {
            const f32 x = input[i] + bias;
            if (x < 0.f) input[i] = 0.01f * x;
            else input[i] = x;
        }
    };

    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) { // post-bias pre-activation?
        if(x[index] >= 0.f) return 1.f;
        return 0.01f;
    }
};

// ----- vlrelu ----- //

class vlrelu : public activation_base {
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) {
        for (u32 i = 0; i < size; i++) {
            const f32 x = input[i] + biases[i];
            if (x < 0.f) input[i] = 0.3f * x;
            else input[i] = x;
        }
    }

    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) {
        for (u32 i = 0; i < size; i++) {
            const f32 x = input[i] + bias;
            if (x < 0.f) input[i] = 0.3f * x;
            else input[i] = x;
        }
    };

    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) { // post-bias pre-activation?
        if (x[index] >= 0.f) return 1.f;
        return 0.3f;
    }
};

// ----- sigmoid ----- //

class sigmoid : public activation_base { 
public:
    virtual void activation([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 *biases, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) 
            input[i] = 1.f / (1.f + std::exp(-(input[i] + biases[i])));
    }

    virtual void activation_c([[maybe_unused]] f32 *input, [[maybe_unused]] const f32 bias, [[maybe_unused]] const u32 size) {
        for(u32 i = 0; i < size; i++) 
            input[i] = 1.f / (1.f + std::exp(-(input[i] + bias)));
    }

    virtual f32 activation_d([[maybe_unused]] const f32 *x, [[maybe_unused]] const u32 index) { // post-activation?
        return (1.f - x[index]) * x[index];
    }
};

activation_base *create_activation(const std::string name) {
    if(name == "null") return new null();
    else if(name == "identity") return new identity();
    else if(name == "tanh") return new tanh();
    else if(name == "elu") return new elu();
    else if(name == "relu") return new relu();
    else if(name == "lrelu") return new lrelu();
    else if(name == "vlrelu") return new vlrelu();
    else if(name == "sigmoid") return new sigmoid();
    else throw std::runtime_error("invalid activation-function '" + name + "'");
}

} // namespace pinguml
