#pragma once

namespace lmj {
struct point {
    long double x{}, y{};
};

// lambda its constructed with should take a "auto &&self" as first parameter for recursion
template<class T>
class recursive_lambda {
    T m_lambda;

public:
    template<class G>
    constexpr recursive_lambda(G &&l) : m_lambda{std::forward<G>(l)} {}

    template<class... Args>
    constexpr decltype(auto) operator()(Args &&... args) {
        return m_lambda(*this, std::forward<Args>(args)...);
    }

    template<class... Args>
    constexpr decltype(auto) operator()(Args &&... args) const {
        return m_lambda(*this, std::forward<Args>(args)...);
    }
};

template<class G>
recursive_lambda(G) -> recursive_lambda<std::decay_t<G>>;
}