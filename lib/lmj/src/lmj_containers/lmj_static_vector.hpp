#pragma once

#include <numeric>
#include <cassert>
#include <limits>
#include <array>
#include <cstdint>
#include "lmj_container_helpers.hpp"

namespace lmj {
template<class... Args>
struct first_type;

template<class T, class... Args>
struct first_type<T, Args...> {
    using type = T;
};

template<class... Args>
using first_type_t = typename first_type<Args...>::type;

template<class... Args>
constexpr auto all_same_types() {
    return !sizeof...(Args) || (... && std::is_same_v<Args, first_type_t<Args...>>);
}

template<class T, std::size_t _capacity>
class static_vector {
public:
    using size_type = decltype(detail::needed_uint<_capacity>());

    T m_data[_capacity]{};
    size_type m_size{};

    constexpr static_vector() = default;

    constexpr explicit static_vector(size_type n) {
        assert(n <= _capacity);
        m_size = n;
    }

    constexpr explicit static_vector(size_type n, T const &value) {
        m_size = n;
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = value;
    }

    template<class Iter>
    constexpr explicit static_vector(Iter begin, Iter end) {
        static_assert(decltype(++begin, begin != end, *begin, 0)() == 0);
        while (begin != end) {
            emplace_back(*begin);
            ++begin;
        }
    }

    constexpr static_vector(static_vector const &) = default;

    constexpr static_vector(static_vector &&) noexcept = default;


    template<std::size_t other_capacity>
    constexpr auto &operator=(static_vector<T, other_capacity> const &other) {
        assert(other.m_size <= _capacity);
        m_size = other.m_size;
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = other.m_data[i];
        return *this;
    }

    template<std::size_t other_capacity>
    constexpr auto &operator=(static_vector<T, other_capacity> &&other) {
        assert(other.m_size <= _capacity);
        m_size = other.m_size;
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = std::move(other.m_data[i]);
        return *this;
    }

    template<std::size_t other_capacity>
    constexpr explicit static_vector(static_vector<T, other_capacity> const &other) {
        *this = other;
    }

    template<std::size_t other_capacity>
    constexpr explicit static_vector(static_vector<T, other_capacity> &&other) noexcept {
        *this = std::move(other);
    }

    constexpr static_vector(std::initializer_list<T> il) {
        assert(il.size() <= _capacity);
        m_size = il.size();
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = il.begin()[i];
    }

    [[nodiscard]] constexpr std::size_t size() const {
        return m_size;
    }

    [[nodiscard]] constexpr std::size_t capacity() const {
        return _capacity;
    }

    template<class G>
    constexpr auto &push_back(G &&elem) {
        return emplace_back(std::forward<G>(elem));
    }

    template<class...Args>
    constexpr auto &emplace_back(Args &&... args) {
        assert(m_size < _capacity && "out of space in static_vector");
        m_data[m_size++] = T(std::forward<Args>(args)...);
        return m_data[m_size - 1];
    }

    [[nodiscard]] constexpr auto &operator[](size_type idx) {
        return m_data[idx];
    }

    constexpr auto const &operator[](size_type idx) const {
        assert(idx < m_size && "no element to return");
        return m_data[idx];
    }

    [[nodiscard]] constexpr auto &front() {
        assert(m_size && "no element to return");
        return m_data[0];
    }

    [[nodiscard]] constexpr auto const &front() const {
        assert(m_size && "no element to return");
        return m_data[0];
    }

    [[nodiscard]] constexpr auto &back() {
        assert(m_size && "no element to return");
        return m_data[m_size - 1];
    }

    [[nodiscard]] constexpr auto const &back() const {
        assert(m_size && "no element to return");
        return m_data[m_size - 1];
    }

    constexpr void pop_back() {
        assert(m_size && "no element to pop");
        m_data[--m_size].~T();
    }

    constexpr void clear() {
        for (size_type i = 0; i < m_size; ++i)
            m_data[i].~T();
        m_size = 0;
    }

    [[nodiscard]] constexpr bool empty() {
        return !m_size;
    }

    [[nodiscard]] constexpr auto begin() const {
        return m_data;
    }

    [[nodiscard]] constexpr auto end() const {
        return m_data + m_size;
    }

    [[nodiscard]] constexpr auto begin() {
        return m_data;
    }

    [[nodiscard]] constexpr auto end() {
        return m_data + m_size;
    }

    [[nodiscard]] constexpr auto cbegin() const {
        return m_data;
    }

    [[nodiscard]] constexpr auto cend() const {
        return m_data + m_size;
    }

    [[nodiscard]] constexpr auto rbegin() const {
        return std::reverse_iterator{end()};
    }

    [[nodiscard]] constexpr auto rend() const {
        return std::reverse_iterator{begin()};
    }

    [[nodiscard]] constexpr auto rbegin() {
        return std::reverse_iterator{end()};
    }

    [[nodiscard]] constexpr auto rend() {
        return std::reverse_iterator{begin()};
    }

    [[nodiscard]] constexpr auto crbegin() const {
        return std::reverse_iterator{end()};
    }

    [[nodiscard]] constexpr auto crend() const {
        return std::reverse_iterator{begin()};
    }

    template<std::size_t other_capacity>
    constexpr auto operator==(static_vector<T, other_capacity> const &other) const {
        if (m_size != other.m_size)
            return false;
        for (size_type i = 0; i < m_size; ++i)
            if (m_data[i] != other.m_data[i])
                return false;
        return true;
    }

    template<class G>
    constexpr auto operator!=(G const &other) {
        return !(*this == other);
    }
};

template<class... Args>
constexpr auto make_static_vector(Args &&... args) {
    return static_vector<std::remove_cvref_t<first_type_t<Args...>>, sizeof...(Args)>{std::forward<Args>(args)...};
}

// testing
static_assert(std::is_same_v<static_vector<int, 3>, decltype(make_static_vector(1, 2, 3))>);
static_assert([] {
    static_vector<int, 1> v;
    return v.push_back(1) == 1;
}());
static_assert([] {
    static_vector<int, 1> a;
    a.push_back(1);
    static_vector<int, 2> b{1};
    return a == b;
}());
static_assert([] {
    return static_vector<int, 2>{1} == static_vector<int, 1>{1};
}());
static_assert([] {
    static_vector<int, 3> v = {1, 2, 3};
    std::array<int, 3> arr1{v[2], v[1], v[0]}, arr2{};
    auto it = v.rbegin();
    for (auto &i: arr2)
        i = *it++;
    return arr1 == arr2;
}());
}

