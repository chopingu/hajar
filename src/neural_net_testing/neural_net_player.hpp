#include "defines.hpp"
#include "neural_net.hpp"
#include "board.hpp"
#include <memory>

namespace gya {
template<class T, class F1, class F2, u64... sizes>
struct neural_net_params {
    using neural_net_t = neural_net<true, false, T, F1, F2, sizes...>;
    using layer_array_t = layer_array<T, sizes...>;
    using weight_array_t = weight_array<T, sizes...>;
};

constexpr static auto fast_activation_function = [](f32 x) {
    return std::clamp(x * 0.2f + 0.5f, 0.0f, 1.0f);
};

constexpr static auto fast_activation_derivative = [](f32 x) {
    if (x < 2.5f) return 0.0f;
    if (x > 2.5f) return 0.0f;
    return 0.2f;
};

constexpr static auto tanh_activation_function = [](f32 x) {
    return std::tanh(x);
};

constexpr static auto tanh_activation_derivative = [](f32 x) {
    return 1.0f - x * x;
};

template<class F1 = decltype(fast_activation_function), class F2 = decltype(fast_activation_derivative)>
struct neural_net_player {
    using neural_net_params_t = neural_net_params<f32, F1, F2, 42, 128, 128, 7>;
    using neural_net_t = typename neural_net_params_t::neural_net_t;
    using layer_array_t = typename neural_net_params_t::layer_array_t;
    using weight_array_t = typename neural_net_params_t::weight_array_t;

    neural_net_t net;

    neural_net_player() : net{F1{}, F2{}} {
        net.update_randomly(0.5);
    }

    neural_net_player(F1 f, F2 derivative) : net{f, derivative} {
        net.update_randomly(0.5);
    }

    auto &operator=(neural_net_player const &other) {
        net = other.net;
        return *this;
    }

    u64 size() const {
        return net.m_weights.data.size() + net.m_biases.data.size();
    }

    [[nodiscard]] u8 operator()(gya::board const &b) {
        if (std::all_of(b.data.begin(), b.data.end(), [](auto &x) { return x.height == 6; }))
            throw std::runtime_error("board is full");

        std::array<f32, 42> input{};
        for (u64 i = 0; i < 6; ++i) {
            for (u64 j = 0; j < 7; ++j) {
                input[i * 7 + j] = b.data[i][j] * b.turn();
            }
        }
        const auto net_output = net.evaluate_const(input);
        u8 ans = 0;
        for (u8 i = 0; i < 7; ++i)
            if (b[i].height < 6 && (b[ans].height >= 6 || net_output[i] > net_output[ans]))
                ans = i;
        return ans;
    }
};
}