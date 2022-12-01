#pragma once

#include "../../include.hpp"

#include "../tensor.hpp"
#include "base.hpp"
#include "math.hpp"

namespace pinguml {

template<class ACTIVATION, class ACTIVATION_C, class ACTIVATION_D>
class fully_connected_layer : public base<ACTIVATION, ACTIVATION_C, ACTIVATION_D> {
public:
    using base_t = base<ACTIVATION, ACTIVATION_C, ACTIVATION_D>; 

    fully_connected_layer(u32 size) : base_t(size, 1, 1), m_biases(1) {
        bias = tensor(m_nodes.rows, m_nodes.cols, m_nodes.channels);
        bias.fill(0.f);
    }

    virtual void push_forward(const base_layer &left_layer, const tensor &weights) {
        const u32 size = weights.m_cols;
        const u32 left_mem_size = left_layer.nodes.size();
        const u32 left_size = left_layer.nodes.rows * left_layer.nodes.cols;

        if(left_layer.m_channel_stride != left_size) {
            for(u32 i = 0; i < size; i++) {
                for(u32 j = 0; j < left_layer.nodes.channels; j++) {
                    nodes.ptr[i] += dot(left_layer.ptr + left_layer.stride
                }
            }
        }
    }
};

} // namespace pinguml  
