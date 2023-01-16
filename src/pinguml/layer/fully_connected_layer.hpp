#pragma once

#include "../../include.hpp"

#include "../utils/tensor.hpp"
#include "../utils/math.hpp"
#include "../utils/activation.hpp"

#include "layer_base.hpp"

namespace pinguml {

class fully_connected_layer : public layer_base { 
public:
    fully_connected_layer(const std::string name, u32 size, activation_base *f) : layer_base(name, size, 1, 1) {
        m_f = f;
        m_uses_biases = 1;
        m_biases = tensor(m_nodes.m_rows, m_nodes.m_cols, m_nodes.m_channels);
        m_biases.fill(0.f);
    }

    virtual std::string config_string() { 
        return "fully_connected " + std::to_string(m_nodes.size()) + " " + m_f->m_name + "\n";
    }

    virtual void push_forward(const layer_base &left_layer, const tensor &weights, [[maybe_unused]] const i32 train = 0) {
        const u32 channel_size = left_layer.m_nodes.m_rows * left_layer.m_nodes.m_cols;
        if (left_layer.m_nodes.m_channel_stride != channel_size) {
            for (u32 i = 0; i < weights.m_rows; i++) 
                for (u32 j = 0; j < left_layer.m_nodes.m_channels; j++) 
                    m_nodes.m_ptr[i] += dot(weights.m_ptr + i * weights.m_cols + j * channel_size, left_layer.m_nodes.m_ptr + j * left_layer.m_nodes.m_channel_stride, channel_size);
        }
        else {
            for(u32 i = 0; i < weights.m_rows; i++)   
                m_nodes.m_ptr[i] += dot(weights.m_ptr + i * weights.m_cols, left_layer.m_nodes.m_ptr, left_layer.m_nodes.size());
        }
    }

    virtual void update_biases(const tensor &delta_biases, const f32 alpha) {
        for(u32 i = 0; i < delta_biases.size(); i++) 
            m_biases.m_ptr[i] -= delta_biases.m_ptr[i] * alpha; // ADD SIMD
    }

    virtual void propogate_delta(layer_base &left_layer, const tensor &weights, [[maybe_unused]] const i32 train = 1) {
        const u32 channel_size = left_layer.m_delta.m_cols * left_layer.m_delta.m_rows;
        if(channel_size != left_layer.m_delta.m_channel_stride) {
            for(u32 i = 0; i < m_delta.size(); i++) {
                const f32 x = m_delta.m_ptr[i];
                for(u32 j = 0; j < left_layer.m_delta.m_channels; j++) 
                    for(u32 k = 0; k < channel_size; k++) 
                        left_layer.m_delta.m_ptr[j * left_layer.m_delta.m_channel_stride + k] += x * weights.m_ptr[i * weights.m_cols + j * channel_size + k];
            }
        }
        else {
            for(u32 i = 0; i < m_delta.size(); i++) {
                const f32 x = m_delta.m_ptr[i];
                for(u32 j = 0; j < left_layer.m_delta.size(); j++) 
                    left_layer.m_delta.m_ptr[j] += x * weights.m_ptr[i * weights.m_cols + j];
            }
        }
    }

    virtual void calculate_delta_weights(const layer_base &left_layer, tensor &delta_weights, [[maybe_unused]] const i32 train = 1) {
        const u32 left_size = left_layer.m_nodes.m_rows * left_layer.m_nodes.m_cols * left_layer.m_nodes.m_channels;
        const u32 delta_size = m_delta.size();
        delta_weights.resize(delta_size, left_size, 1);

        const u32 channel_size = left_layer.m_nodes.m_rows * left_layer.m_nodes.m_cols;
        for(u32 i = 0; i < delta_size; i++) {
            const f32 x = m_delta.m_ptr[i];

            if(left_size != left_layer.m_nodes.size()) {
                for(u32 j = 0; j < left_layer.m_nodes.m_channels; j++) 
                    for(u32 k = 0; k < channel_size; k++) 
                        delta_weights.m_ptr[i * left_size + j * channel_size + k] = x * left_layer.m_nodes.m_ptr[j * left_layer.m_nodes.m_channel_stride + k];
            }
            else {
                for(u32 j = 0; j < left_size; j++) 
                    delta_weights.m_ptr[i * left_size + j] = x * left_layer.m_nodes.m_ptr[j];
            }
        }
    }
};

} // namespace pinguml  
