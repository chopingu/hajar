#pragma once

#include "../include.hpp"
#include "neural_net.hpp"
#include "neural_net_player.hpp"

namespace gya {

template<class F1 = decltype(fast_activation_function), class F2 = decltype(fast_activation_derivative)>
struct neural_net_player_test {
    using neural_net_params_t = neural_net_params<f32, F1, F2, 42, 128, 128, 42>;
    using neural_net_t = typename neural_net_params_t::neural_net_t;
    using layer_array_t = typename neural_net_params_t::layer_array_t;
    using weight_array_t = typename neural_net_params_t::weight_array_t;

    neural_net_t net;

    neural_net_player_test() : net{F1{}, F2{}} {
        net.update_randomly(0.5);
    }

    neural_net_player_test(F1 f, F2 derivative) : net{f, derivative} {
        net.update_randomly(0.5);
    }

    auto &operator=(neural_net_player_test const &other) {
        net = other.net;
        return *this;
    }

    [[nodiscard]] u64 size() const {
        return net.m_weights.data.size() + net.m_biases.data.size();
    }

    [[nodiscard]] u8 operator()(gya::board const &b) {
        if (std::all_of(b.data.begin(), b.data.end(), [](auto &x) { return x.height == 6; }))
            throw std::runtime_error("board is full");

#define MATRIX(x) (*reinterpret_cast<std::array<std::array<f32, 7>, 6>*>(&x))
        std::array<f32, 42> input{};
        for (u64 i = 0; i < 6; ++i) {
            for (u64 j = 0; j < 7; ++j) {
                // reinterpret casting to avoid copying
                MATRIX(input)[i][j] = static_cast<f32>(b[i][j] * b.turn());
            }
        }
        auto net_output = net.evaluate_const(input);
        // reinterpret casting to avoid copying
        auto &outp = MATRIX(net_output);
#undef MATRIX

        u8 ans = 0;
        f32 ans_val = -1e9;
        for (u8 i = 0; i < 7; ++i) {
            if (b[i].height >= 6)
                continue;
            f32 val = outp[i][b[i].height];
            if (val > ans_val || b[ans].height == 6){
                ans = i;
                ans_val = val;
            }
        }
        return ans;
    }
};
}
