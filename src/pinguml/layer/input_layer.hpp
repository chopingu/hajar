#pragma once

#include "../../include.hpp"

#include "../utils/tensor.hpp"
#include "../utils/activation.hpp"

#include "layer_base.hpp"

namespace pinguml {

class input_layer : public layer_base {
public:
    input_layer(const std::string name, const u32 h, const u32 w, const u32 c) : layer_base(name, h, w, c) {
        m_f = create_activation("identity");
    }

    virtual ~input_layer() {}

    virtual void activate() {}

    virtual std::string config_string() {
        return "input " + std::to_string(m_nodes.m_rows) + " " + std::to_string(m_nodes.m_cols) + " " + std::to_string(m_nodes.m_channels) + " " + m_f->m_name + "\n"; 
    }
};

} // namespace pinguml
