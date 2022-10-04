#pragma once

#include <cmath>
#include <string>

/*
 * usage:
 * new_cost_function(name_of_function) 
 * creates a pointer to a struct with the cost function, 
 * the derivative of the partial derivative of the cost function
 * and a string that contains the name of the function
*/
namespace cnn {

namespace mse {
constexpr auto name = "mse";

template<class T>
T cost(T output, T target) { return 0.5f * (output - target) * (output - target); }

template<class T>
T d_cost(T output, T target) { return output - target; }
}

namespace bce {
constexpr auto name = "bce";

template<class T>
T cost(T output, T target) { return (-target * std::log(output) - (1.0f - target) * std::log(1.0f - output)); }

template<class T>
T d_cost(T output, T target) { return (output - target) / (output * (1.0f - output)); }
}

template<class T>
struct cost_function {
    const char *name;
    using function_ptr = T(*)(T, T);

    function_ptr cost, d_cost;
};

template<class T>
cost_function<T> *new_cost_function(std::string_view new_function) {
    cost_function<T> *nw = new cost_function<T>;

    if (new_function == mse::name) {
        nw->cost = &mse::cost;
        nw->d_cost = &mse::d_cost;
        nw->name = mse::name;
        return nw;
    }

    if (new_function == bce::name) {
        nw->cost = &bce::cost;
        nw->d_cost = &bce::d_cost;
        nw->name = bce::name;
    }

    delete nw;
    return nullptr;
}
}
