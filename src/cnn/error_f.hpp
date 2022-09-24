#pragma once

#include <cmath>
#include <string>
#include "../defines.hpp"

namespace gya {

namespace cnn 
{

template<class T>

namespace mse 
{
std::string name="mse";
T error(T output, T target) { return 0.5 * (output - target) * (output - target); }
T pd_error(T output, T target) { return out - target; }
}

namespace bce
{
std::string name="bce";
T error(T output, T target) { return (-target * std::log(output) - (1.0 - target) * std::log(1.0 - output)); }
T pd_error(T output, T target) { return (output - target) / (output * (1.0 - output)); }
}

struct error_function {
    std::string *name;
    T (*error)(T, T);
    T (*pd_error)(T, T);
}

error_function* new_error_function(std::string new_function) {
    error_function *nw = new error_function;

    if(new_function==mse::name) { 
        nw->error=&mse::error;
        nw->pd_error=&mse::pd_error;
        nw->name=mse::name;
        return nw;
    }

    if(new_function==bce::name) {
        nw->error=&bce::error;
        nw->pd_error=&bce::pd_error;
        nw->name=bce::name;
    }

    delete nw;
    return 0;
}

}
