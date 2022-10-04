#pragma once

#include <cmath>
#include <array>
#include "../defines.hpp"

namespace cnn {
template<class T>
T dot(const T *v1, const T *v2, const u64 _size) { 
    T sum = 0;
    for(u64 i = 0; i < _size; i++) 
        sum += v1[i] * v2[i];

    return sum;
}

template<class T> 
T dot_2d(const T *v1, const T *v2, const u64 _size, const u64 stride1, const u64 stride2) {
    T sum = 0;
    for(u64 i = 0; i < _size; i++) 
        sum += dot(v1[stride1 * i], v2[stride2 * i], _size);

    return sum;
}

template<class T> 
T dot_180(const T *v1, const T *v2, const u64 _size) { 
    T sum = 0;
    for(u64 i = 0; i < _size; i++) 
        sum += v1[i] * v2[_size - 1 - i];

    return sum;
}
}
