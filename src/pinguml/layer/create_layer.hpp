#pragma once

#include "../../include.hpp"

#include "../utils/activation.hpp"

#include "layer_base.hpp"
#include "input_layer.hpp"
#include "fully_connected_layer.hpp"

namespace pinguml {

layer_base *create_layer(const std::string name, const std::string build) {
    std::istringstream str(build);

    u32 height, width, channels;
    std::string layer_type; 
    str >> layer_type;

    if(layer_type == "input") {
        str >> height >> width >> channels;
        return new input_layer(name, height, width, channels);
    }
    else if(layer_type == "fully_connected") {
        std::string activation;
        str >> height >> activation;

        if(height <= 0) 
            throw std::runtime_error("invalid number of nodes in fully connected layer");

        return new fully_connected_layer(name, height, create_activation(activation));
    }
    else throw std::runtime_error("invalid layer-type '" + layer_type + "'");

    return nullptr;
}

} // namespace pinguml
