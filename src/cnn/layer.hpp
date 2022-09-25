#pragma once

#include "activation.hpp"

namespace gya {

namespace cnn
{

template<class T>

class base_layer {
private:
    T learning_rate;
public:
    activation_function *act_f;
      
    void set_learning_rate(T alpha = 1.0f) { learning_rate = 1.0f; }

    
}

}

}   
