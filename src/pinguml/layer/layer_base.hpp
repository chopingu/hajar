#pragma once

#include "../../include.hpp"

#include "../utils/tensor.hpp"
#include "../utils/activation.hpp"

namespace pinguml {

class layer_base {
protected:
    bool m_uses_weights;
    bool m_uses_biases;
    f32 m_learning_rate;

public:
    std::string m_name;

    activation_base *m_f;

    tensor m_nodes;
    tensor m_biases;
    tensor m_delta;

    u32 m_pad_rows;
    u32 m_pad_cols;

    std::vector<std::pair<u32, layer_base*>> m_forward_connections;
    std::vector<std::pair<u32, layer_base*>> m_backward_connections;

    layer_base(const std::string name, const u32 h, const u32 w, const u32 c) : m_uses_weights(1), m_uses_biases(0), m_learning_rate(1.f), m_name(name), m_nodes(h, w, c), m_delta(h, w, c), m_pad_rows(0), m_pad_cols(0) {}

    virtual ~layer_base() {}

    virtual bool uses_weights() {
        return m_uses_weights;
    }

    virtual bool uses_biases() {
        return m_uses_biases;
    }

    virtual tensor *create_connection(layer_base &left_layer, const u32 index) {
        left_layer.m_forward_connections.push_back({index, this});
        m_backward_connections.push_back({index, &left_layer});
        if (!m_uses_weights) return nullptr;
        u32 rows = left_layer.m_nodes.m_rows * left_layer.m_nodes.m_cols * left_layer.m_nodes.m_channels;
        u32 cols = m_nodes.m_rows * m_nodes.m_cols * m_nodes.m_channels;
        return new tensor(rows, cols, 1);
    }

    virtual void activate() {
        if (m_uses_biases)
            m_f->activation(m_nodes.m_ptr, m_biases.m_ptr,  m_nodes.size());
        else
            m_f->activation_c(m_nodes.m_ptr, 0.f, m_nodes.size());
    }

    void set_learning_rate(const f32 alpha) { m_learning_rate = alpha; }

    virtual void push_forward([[maybe_unused]] const layer_base &left_layer, [[maybe_unused]] const tensor &weights, [[maybe_unused]] i32 train) {}

    f32 df(f32 *in, u32 i) { 
        if (m_f) return m_f->activation_d(in, i); 
        return 1.f;
    }

    virtual std::string config_string() { return ""; }

    virtual void calculate_delta_weights([[maybe_unused]] const layer_base &left_layer, [[maybe_unused]] tensor &delta_weights) {}

    virtual void propogate_delta([[maybe_unused]] layer_base &left_layer, [[maybe_unused]] const tensor &weights) {}

    virtual void update_biases([[maybe_unused]] const tensor &delta_biases, [[maybe_unused]] const f32 alpha) {}
};

} // namespace pinguml
