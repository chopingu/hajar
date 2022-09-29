#pragma once

#include <cmath>
#include <array>
#include "../defines.hpp"

namespace gya {

namespace cnn 
{ 

template<class T>

T v_dot(std::span<const T> v1, std::span<const T> v2, const u64 _size, const u64 _start) { 
    T sum = 0;
    for(u64 i = 0; i < _size; i++) 
        sum += v1[_start + i] * v2[_start + i];
    return sum;
}

}

}

