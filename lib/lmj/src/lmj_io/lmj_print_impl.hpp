#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "../lmj_utils/lmj_utils.hpp"

namespace lmj {
using std::operator ""s;

template<class T>
concept print_stream = requires(T x) {
    x << int{};
    x << double{};
    x << "";
    x << ""s;
};

template<class T>
concept printable = requires(T x) {
    std::cout << x;
};

constexpr void print_impl_pretty(print_stream auto &out, printable auto &&x) {
    out << x;
}

template<class T>
requires iterable<T> && (!std::is_convertible_v<T, std::string>)
constexpr void print_impl_pretty(print_stream auto &out, T &&x);

template<class T, class G>
constexpr void print_impl_pretty(print_stream auto &out, std::pair<T, G> const &p) {
    out << '{';
    print_impl_pretty(out, p.first);
    out << ", ";
    print_impl_pretty(out, p.second);
    out << '}';
}

template<class T>
requires iterable<T> && (!std::is_convertible_v<T, std::string>)
constexpr void print_impl_pretty(print_stream auto &out, T &&x) {
    out << '{';
    auto first = true;
    for (auto &&i: x) {
        if (!first) {
            if constexpr (iterable<decltype(i)>)
                out << ",\n";
            else
                out << ", ";
        }
        first = false;
        print_impl_pretty(out, i);
    }
    out << '}';
}


constexpr void print_impl(print_stream auto &out, printable auto &&x) {
    out << x;
}

template<class T>
requires iterable<T> && (!std::is_convertible_v<T, std::string>)
constexpr void print_impl(print_stream auto &out, T &&x);

template<class T, class G>
constexpr void print_impl(print_stream auto &out, std::pair<T, G> const &p) {
    print_impl(out, p.first);
    out << ' ';
    print_impl(out, p.second);
}

template<class T>
requires iterable<T> && (!std::is_convertible_v<T, std::string>)
constexpr void print_impl(print_stream auto &out, T &&x) {
    auto first = true;
    for (auto &&i: x) {
        if (!first)
            out << ' ';
        first = false;
        print_impl(out, i);
    }
}
}