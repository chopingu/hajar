#pragma once

#include <iostream>
#include "print_impl.hpp"
#include "fast_print_impl.hpp"

#ifndef USE_FAST_IO
#define USE_FAST_IO 1
#endif

namespace lmj {
#if USE_FAST_IO

inline void print(auto &&x) {
    fast_print::print_impl(stdout, x);
    fast_print::print_impl(stdout, '\n');
}

inline void print(auto &&x, auto &&...pack) {
    fast_print::print_impl(stdout, x);
    fast_print::print_impl(stdout, ' ');
    print(pack...);
}

#else
inline void print(auto &&x) {
    print_impl(std::cout, x);
    print_impl(std::cout, '\n');
}

inline void print(auto &&x, auto &&...pack) {
    print_impl(std::cout, x);
    print_impl(std::cout, ' ');
    print(pack...);
}
#endif
}
