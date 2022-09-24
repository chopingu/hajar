#pragma once

#include <cmath>
#include <string>
#include <../defines.hpp>

namespace gya {

namespace cnn 
{

namespace mse 
{
string name="mse";
f32 error(f32 output, f32 target) { return 0.5f * (output - target) * (output - target); }
f32 pd_error(f32 output, f32 target) { return out - target; }
}

namespace bce
{
string name="bce";
f32 error(f32 output, f32 target) { return (-target * std::log(output) - (1.0f - target) * std::log(1.0f - output)); }
f32 pd_error(f32 output, f32 target) { return (output - target) / (output * (1.0f - output)); }
}

struct error_function {
    string *name;
    f32 (*error)(f32, f32);
    f32 (*pd_error)(f32, f32);
}

error_function* new_error_function(string new_function) {
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
