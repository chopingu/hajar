#pragma once

#include "../include.hpp"
#include "neural_net.hpp"
#include "neural_net_player.hpp"

namespace gya {

template<class F1 = decltype(tanh_activation_function),
         class F2 = decltype(tanh_activation_derivative)>
struct neural_net_player_deep {
    template<class T, class F1_, class F2_, usize... sizes>
    struct neural_net_params {
        using neural_net_t = neural_net<false, false, T, F1_, F2_, sizes...>;
        using layer_array_t = layer_array<T, sizes...>;
        using weight_array_t = weight_array<T, sizes...>;
    };

    using move_choosing_net_t = neural_net_params<f32, F1, F2, 42, 128, 64, 32, 16, 8, 7>;
    using move_neural_net_t = typename move_choosing_net_t::neural_net_t;
    using move_layer_array_t = typename move_choosing_net_t::layer_array_t;
    using move_weight_array_t = typename move_choosing_net_t::weight_array_t;

    using state_choosing_net_t = neural_net_params<f32, F1, F2, 42, 128, 64, 32, 1>;
    using state_neural_net_t = typename state_choosing_net_t::neural_net_t;
    using state_layer_array_t = typename state_choosing_net_t::layer_array_t;
    using state_weight_array_t = typename state_choosing_net_t::weight_array_t;


    move_neural_net_t m_move_net;
    state_neural_net_t m_state_net;

    struct move_state {
        gya::board board;
        u8 played_move;
    };

    std::vector<move_state> m_prev_states;

    neural_net_player_deep() : m_move_net{F1{}, F2{}} { m_move_net.update_randomly(0.5); }

    neural_net_player_deep(F1 f, F2 derivative) : m_move_net{f, derivative} { m_move_net.update_randomly(0.5); }

    auto &operator=(neural_net_player_deep const &other) {
        m_move_net = other.m_move_net;
        return *this;
    }

    [[nodiscard]] usize size() const {
        return m_move_net.m_weights.data.size() + m_move_net.m_biases.data.size();
    }

    // do the Q-learning stuff
    void learn(bool won, bool lost) {
        // reward for last state
        f32 reward = won - lost;

        constexpr f32 learning_rate = 0.01f, discount_factor = 0.9f;

        // temporal difference for last move
        const auto last_board = m_prev_states.back().board;
        const auto last_outp = m_move_net.evaluate(last_board);
        const auto last_move = m_prev_states.back().move;

        const f32 old_last_q = m_state_net.evaluate(last_board);
        const f32 last_q = old_last_q + learning_rate * (reward - old_last_q);

        std::array<f32, gya::BOARD_WIDTH> correct_last_output = last_outp.back();
        correct_last_output[last_move] += last_q;

        m_move_net.apply_derivatives(m_move_net.compute_derivatives(correct_last_output));

        f32 next_q_value = last_q;

        //  go through all layers except the last one
        for (usize i = m_prev_states.size() - 1; i-- > 0;) {
            std::array move_output = m_move_net.evaluate(m_prev_states[i].board);
            const f32 state_output = m_state_net.evaluate(m_prev_states[i].board);
            const f32 q_state = state_output + learning_rate * (discount_factor * next_q_value - state_output);
            const f32 q_move = move_output + learning_rate * (discount_factor * next_q_value - move_output);

            const std::array<f32, 1> temp{q_state};
            m_state_net.apply_derivatives(m_state_net.compute_derivatives(temp, false));

            move_output[m_prev_states[i].move] = q_move;
            m_move_net.apply_derivatives(m_move_net.compute_derivatives(move_output, false));

            next_q_value = state_output;
        }
    }

    [[nodiscard]] u8 operator()(gya::board const &b) {
        if (std::all_of(b.data.begin(), b.data.end(), [](auto &x) { return x.height == gya::BOARD_HEIGHT; }))
            throw std::runtime_error("board is full");

        std::array<f32, gya::BOARD_WIDTH * gya::BOARD_HEIGHT> input{};
        for (usize i = 0; i < gya::BOARD_HEIGHT; ++i) {
            for (usize j = 0; j < gya::BOARD_WIDTH; ++j) {
                input[gya::BOARD_WIDTH * i + j] = static_cast<f32>(b[i][j] * b.turn());
            }
        }

        auto net_output = m_move_net.evaluate(input);

        u8 ans = 0;
        f32 ans_val = -std::numeric_limits<f32>::max();
        for (u8 i = 0; i < gya::BOARD_WIDTH; ++i) {
            if (b[i].height == gya::BOARD_HEIGHT) continue;
            f32 val = net_output[gya::BOARD_WIDTH * i + b[i].height];
            if (val > ans_val || b[ans].height == gya::BOARD_HEIGHT) {
                ans = i;
                ans_val = val;
            }
        }

        // store all previous states of the network for Q-learning
        m_prev_states.emplace_back(m_move_net.m_values, ans);

        return ans;
    }
};
} // namespace gya
