#pragma once

#include "../include.hpp"
#include "neural_net.hpp"
#include "../board.hpp"

namespace gya {
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
    template<class T, class F1_, class F2_, usize... sizes>
    struct neural_net_params {
        using neural_net_t = neural_net<false, false, T, F1_, F2_, sizes...>;
        using layer_array_t = layer_array<T, sizes...>;
        using weight_array_t = weight_array<T, sizes...>;
    };

    using neural_net_params_t = neural_net_params<f32, F1, F2, 42, 35, 28, 21, 14, 7>;
    using neural_net_t = typename neural_net_params_t::neural_net_t;
    using layer_array_t = typename neural_net_params_t::layer_array_t;
    using weight_array_t = typename neural_net_params_t::weight_array_t;

    neural_net_t m_net;

    neural_net_player() : m_net{F1{}, F2{}} {
        m_net.update_randomly(0.5);
    }

    neural_net_player(F1 f, F2 derivative) : m_net{f, derivative} {
        m_net.update_randomly(0.5);
    }

    constexpr neural_net_player &operator=(neural_net_player const &other) = default;

    constexpr neural_net_player(neural_net_player const &other) = default;

    [[nodiscard]] constexpr usize size() const {
        return m_net.m_weights.m_data.size() + m_net.m_biases.m_data.size();
    }

    [[nodiscard]] u8 operator()(gya::board const &b) {
        if (std::all_of(b.data.begin(), b.data.end(), [](auto &x) { return x.height == 6; }))
            throw std::runtime_error("board is full");

        std::array<f32, gya::BOARD_WIDTH * gya::BOARD_HEIGHT> input{};
        for (usize i = 0; i < gya::BOARD_HEIGHT; ++i) {
            for (usize j = 0; j < gya::BOARD_WIDTH; ++j) {
                input[i * gya::BOARD_WIDTH + j] = b.data[i][j] * b.turn();
            }
        }
        const auto net_output = m_net.evaluate_const(input);
        u8 ans = 0;
        for (u8 i = 0; i < 7; ++i)
            if (b[i].height < 6 && (b[ans].height >= 6 || net_output[i] > net_output[ans]))
                ans = i;
        return ans;
    }
};
}
