#pragma once

#include "../../include.hpp"

#include "../tensor.hpp"
#include "base.hpp"

namespace pinguml {

template<class ACTIVATION, class ACTIVATION_C, class ACTIVATION_D>
class input_layer : public base<ACTIVATION, ACTIVATION_C, ACTIVATION_D> {
public:
    using base_t = base<ACTIVATION, ACTIVATION_C, ACTIVATION_D>;

    input_layer(const u32 h, const u32 w, const u32 c) : base_t(h, w, c) {}

    virtual ~input_layer() {}
    virtual void activate() {}
    virtual void activate_delta(base_t &left_layer, const tensor &weights) {}
    virtual void calculate_delta(const base_t &left_layer, tensor &delta) {}
    virtual void push_forward(const base_t &left_layer, const tensor &weights) {}
};

} // namespace pinguml
