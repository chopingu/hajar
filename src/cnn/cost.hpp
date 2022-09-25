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

namespace gya {

namespace cnn 
{

template<class T>

namespace mse 
{
std::string name="mse";
T cost(T output, T target) { return 0.5f * (output - target) * (output - target); }
T d_cost(T output, T target) { return out - target; }
}

namespace bce
{
std::string name="bce";
T cost(T output, T target) { return (-target * std::log(output) - (1.0f - target) * std::log(1.0f - output)); }
T d_cost(T output, T target) { return (output - target) / (output * (1.0f - output)); }
}

struct cost_function {
    std::string *name;
    T (*cost)(T, T);
    T (*d_cost)(T, T);
}

cost_function* new_cost_function(std::string new_function) {
    cost_function *nw = new cost_function;

    if(new_function==mse::name) { 
        nw->cost=&mse::cost;
        nw->d_cost=&mse::d_cost;
        nw->name=mse::name;
        return nw;
    }

    if(new_function==bce::name) {
        nw->cost=&bce::cost;
        nw->d_cost=&bce::d_cost;
        nw->name=bce::name;
    }

    delete nw;
    return 0;
}

}

}
