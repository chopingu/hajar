#pragma once

#include "../include.hpp"

namespace pinguml {

f32 dot(const f32 *a, const f32 *b, const u32 size) {
    f32 sum = 0;
    for (u32 i = 0; i < size; i++)
        sum += a[i] * b[i];

    return sum;
}

} // pinguml
