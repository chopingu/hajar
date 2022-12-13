#pragma once

#include <iostream>

#include "print_impl.hpp"
#include "fast_print_impl.hpp"

#ifndef USE_FAST_IO
#define USE_FAST_IO 1
#endif

namespace lmj {
#if USE_FAST_IO

inline void debug(auto &&x) {
    fast_print::print_impl_pretty(stderr, x);
    fast_print::print_impl_pretty(stderr, '\n');
}

inline void debug(auto &&x, auto &&...pack) {
    fast_print::print_impl_pretty(stderr, x);
    fast_print::print_impl_pretty(stderr, ' ');
    debug(pack...);
}

#else
inline void debug(auto &&x) {
    print_impl_pretty(std::cerr, x);
    print_impl_pretty(std::cerr, '\n');
}

inline void debug(auto &&x, auto &&...pack) {
    print_impl_pretty(std::cerr, x);
    print_impl_pretty(std::cerr, ' ');
    debug(pack...);
}
#endif
}
