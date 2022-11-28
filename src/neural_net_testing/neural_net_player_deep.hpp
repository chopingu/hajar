#pragma once

#include "../include.hpp"
#include "neural_net.hpp"
#include "neural_net_player.hpp"

namespace gya {

template<class F1 = decltype(fast_activation_function), class F2 = decltype(fast_activation_derivative)>
struct neural_net_player_deep {
    using neural_net_params_t = neural_net_params<f32, F1, F2, 42, 128, 128, 7>;
    using neural_net_t = typename neural_net_params_t::neural_net_t;
    using layer_array_t = typename neural_net_params_t::layer_array_t;
    using weight_array_t = typename neural_net_params_t::weight_array_t;

    neural_net_t m_net;

    // pairs of states of the network at each move, along with the chosen move
    std::vector<std::pair<layer_array_t, u8>> m_prev_states;

    neural_net_player_deep() : m_net{F1{}, F2{}} {
        m_net.update_randomly(0.5);
    }

    neural_net_player_deep(F1 f, F2 derivative) : m_net{f, derivative} {
        m_net.update_randomly(0.5);
    }

    auto &operator=(neural_net_player_deep const &other) {
        m_net = other.m_net;
        return *this;
    }

    [[nodiscard]] usize size() const {
        return m_net.m_weights.data.size() + m_net.m_biases.data.size();
    }

    // do the Q-learning stuff
    void learn(bool won, bool lost) {
        // reward for last state
        f32 reward = won - lost;

        constexpr f32 learning_rate = 0.01f, discount_factor = 0.9;

        // temporal difference for last move
        const auto &last_outp = m_prev_states.back().first;
        const auto last_move = m_prev_states.back().second;

        const f32 last_tdiff = learning_rate * (reward - last_outp.back()[last_move]);

        std::array<f32, 7> correct = last_outp.back();
        correct[last_move] += last_tdiff;

        m_net.m_values = last_outp;

        for (usize i = m_prev_states.size() - 1; i-- > 0;) {
            std::array<f32, 7> outp = m_prev_states[i].first.back();
            u8 chosen = m_prev_states[i].second;

        }
    }

    [[nodiscard]] u8 operator()(gya::board const &b) {
        if (std::all_of(b.data.begin(), b.data.end(), [](auto &x) { return x.height == 6; }))
            throw std::runtime_error("board is full");

        std::array<f32, 42> input{};
        for (usize i = 0; i < 6; ++i) {
            for (usize j = 0; j < 7; ++j) {
                input[7 * i + j] = static_cast<f32>(b[i][j] * b.turn());
            }
        }

        auto net_output = m_net.evaluate(input);

        u8 ans = 0;
        f32 ans_val = -1e9;
        for (u8 i = 0; i < 7; ++i) {
            if (b[i].height >= 6)
                continue;
            f32 val = net_output[7 * i + b[i].height];
            if (val > ans_val || b[ans].height == 6) {
                ans = i;
                ans_val = val;
            }
        }

        // store all previous states of the network for Q-learning
        m_prev_states.emplace_back(m_net.m_values, ans);

        return ans;
    }
};
}
