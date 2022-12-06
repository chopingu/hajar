#pragma once

#include "../../include.hpp"

#include "../tensor.hpp"

namespace pinguml {

template<class ACTIVATION, class ACTIVATION_C, class ACTIVATION_D>
class base {
private:
    bool m_weights;
    bool m_biases;
    f32 m_learning_rate;
public:
    tensor m_nodes;
    tensor m_bias;
    tensor m_delta;

    u32 m_pad_rows;
    u32 m_pad_cols;

    auto activation(f32 *input, const f32 *biases, const u32 size) {
        return ACTIVATION{}(input, biases, size);
    }

    auto activation_c(f32 *input, const f32 bias, const u32 size) {
        return ACTIVATION_C{}(input, bias, size);
    }

    // auto activation_d(...) {
    //     return ACTIVATION_D{}(...);
    // }

    std::vector<base *> forward_connections;
    std::vector<base *> backward_connections;

    base(const u32 h, const u32 w, const u32 c) : _weights(1), _biases(0), _learning_rate(1.f), nodes(h, w, c),
                                                  delta(h, w, c), pad_rows(0), pad_cols(0) {}

    virtual ~base() {}

    virtual tensor *new_connections(const base &left_layer, const u32 index) {
        left_layer.forward_connections.push_back(this);
        backward_connections.push_back(&left_layer);
        if (!_weights) return nullptr;
        u32 rows = left_layer.nodes.rows * left_layer.nodes.cols * left_layer.nodes.channels;
        u32 cols = nodes.rows * nodes.cols * nodes.channels;
        return new tensor(rows, cols, 1);
    }

    virtual void activate() {
        if (_biases)
            activation(nodes.ptr, nodes.size(), biases.ptr);
        else
            activation(nodes.ptr, nodes.size(), 0.f);
    }

    void learning_rate(f32 alpha) { _learning_rate = alpha; }

    virtual void activate_delta(base &left_layer, const tensor &weights) {}

    virtual void calculate_delta(const base &left_layer, tensor &delta) {}

    virtual void update_bias(const tensor &biases_nw, f32 alpha) {}

    virtual void push_forward(const base &left_layer, const tensor &weights) {}
};

} // namespace pinguml
