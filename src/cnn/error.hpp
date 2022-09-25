#pragma once

#include <cmath>
#include <string>
#include "../defines.hpp"

/*
 * usage:
 * new_error_function(name_of_function) 
 * creates a pointer to a struct with the error function, 
 * the derivative of the partial derivative of the error function
 * and a string that contains the name of the function
*/

namespace gya {

namespace cnn 
{

template<class T>

namespace mse 
{
std::string name="mse";
T error(T output, T target) { return 0.5f * (output - target) * (output - target); }
T d_error(T output, T target) { return out - target; }
}

namespace bce
{
std::string name="bce";
T error(T output, T target) { return (-target * std::log(output) - (1.0f - target) * std::log(1.0f - output)); }
T d_error(T output, T target) { return (output - target) / (output * (1.0f - output)); }
}

struct error_function {
    std::string *name;
    T (*error)(T, T);
    T (*d_error)(T, T);
}

error_function* new_error_function(std::string new_function) {
    error_function *nw = new error_function;

    if(new_function==mse::name) { 
        nw->error=&mse::error;
        nw->d_error=&mse::d_error;
        nw->name=mse::name;
        return nw;
    }

    if(new_function==bce::name) {
        nw->error=&bce::error;
        nw->d_error=&bce::d_error;
        nw->name=bce::name;
    }

    delete nw;
    return 0;
}

}

}
