#pragma once

#include <cstdio>
#include <cstdint>
#include "../utils/utils.hpp"
#include "concepts"

namespace lmj::fast_print {

inline auto print_impl(FILE *fptr, floating_point auto x) {
    std::fprintf(fptr, "%Lf", static_cast<long double>(x));
}

inline auto print_impl(FILE *fptr, unsigned_integral auto x) {
    if constexpr (std::is_same_v<decltype(x), unsigned char> || std::is_same_v<decltype(x), char>) {
        std::fputc(x, fptr);
    } else {
        if constexpr (sizeof(x) <= 8) {
            std::fprintf(fptr, "%llu", static_cast<uint64_t>(x));
        } else {
            char buff[(sizeof(x) * 8 * 10 + 2) / 3]{};
            std::size_t size = 0;
            do {
                buff[size++] = (x % 10) + '0';
                x /= 10;
            } while (x);
            std::reverse(buff, buff + size);
            std::fputs(buff, fptr);
        }
    }
}

inline auto print_impl(FILE *fptr, signed_integral auto x) {
    if constexpr (std::is_same_v<decltype(x), signed char> || std::is_same_v<decltype(x), char>) {
        std::fputc(x, fptr);
    } else {
        if (x < 0) {
            std::fputc('-', fptr);
            if (x == std::numeric_limits<decltype(x)>::min()) {
                print_impl(fptr,
                           static_cast<std::make_unsigned_t<decltype(x)>>(std::numeric_limits<decltype(x)>::max()) + 1);
            } else {
                print_impl(fptr, static_cast<std::make_unsigned_t<decltype(x)>>(-x));
            }
        } else {
            print_impl(fptr, static_cast<std::make_unsigned_t<decltype(x)>>(x));
        }
    }
}

inline auto print_impl(FILE *fptr, std::string_view x) {
    for (auto i: x)
        print_impl(fptr, i);
}

inline auto print_impl_pretty(FILE *fptr, floating_point auto x) {
    print_impl(fptr, x);
}

inline auto print_impl_pretty(FILE *fptr, unsigned_integral auto x) {
    print_impl(fptr, x);
}

inline auto print_impl_pretty(FILE *fptr, signed_integral auto x) {
    print_impl(fptr, x);
}

inline auto print_impl_pretty(FILE *fptr, std::string_view x) {
    print_impl(fptr, '"');
    print_impl(fptr, x);
    print_impl(fptr, '"');
}

template<class T, class G>
auto print_impl(FILE *fptr, std::pair<T, G> const &x);

template<class T>
requires lmj::iterable<T> && (!std::is_convertible_v<T, std::string> && !std::is_convertible_v<T, std::string_view>)
auto print_impl(FILE *fptr, T &&x) {
    bool first = true;
    for (auto &&i: x) {
        if (!first)
            print_impl(fptr, ' ');
        first = false;
        print_impl(fptr, i);
    }
}

template<class T, class G>
auto print_impl(FILE *fptr, std::pair<T, G> const &x) {
    print_impl(fptr, x.first);
    print_impl(fptr, ' ');
    print_impl(fptr, x.second);
}

template<class T, class G>
auto print_impl_pretty(FILE *fptr, std::pair<T, G> const &x);

template<class T>
requires lmj::iterable<T> && (!std::is_convertible_v<T, std::string> && !std::is_convertible_v<T, std::string_view>)
auto print_impl_pretty(FILE *fptr, T &&x) {
    print_impl_pretty(fptr, '{');
    bool first = true;
    for (auto &&i: x) {
        if (!first) {
            if constexpr (iterable<decltype(i)> || requires { i.second; i.first; })
                print_impl(fptr, ",\n");
            else
                print_impl(fptr, ", ");
        }
        first = false;
        print_impl_pretty(fptr, i);
    }
    print_impl_pretty(fptr, '}');
}

template<class T, class G>
auto print_impl_pretty(FILE *fptr, std::pair<T, G> const &x) {
    print_impl_pretty(fptr, '{');
    print_impl_pretty(fptr, x.first);
    print_impl_pretty(fptr, ',');
    print_impl_pretty(fptr, x.second);
    print_impl_pretty(fptr, '}');
}


}
