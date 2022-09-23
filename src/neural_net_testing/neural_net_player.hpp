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
    using neural_net_params_t = neural_net_params<f32, F1, F2, 43, 35, 30, 25, 20, 15, 10, 7>;
    using neural_net_t = typename neural_net_params_t::neural_net_t;
    using layer_array_t = typename neural_net_params_t::layer_array_t;

    neural_net_player() : net{fast_activation_function, fast_activation_derivative} {}

    neural_net_player(F1 f, F2 derivative) : net{f, derivative} {}

    auto &operator=(neural_net_player const &other) {
        net = other.net;
        return *this;
    }

    neural_net_t net;

    u64 size() const {
        return net.m_weights.data.size() + net.m_biases.data.size();
    }

    void train(std::span<f32> correct_output) {

    }

    [[nodiscard]] u8 operator()(gya::board const &b) {
        if (std::all_of(b.data.begin(), b.data.end(), [](auto &x) { return x.height == 6; }))
            throw std::runtime_error("board is full");

        std::array<f32, 43> input{};
        for (u64 i = 0; i < 6; ++i) {
            for (u64 j = 0; j < 7; ++j) {
                input[i * 7 + j] = b.data[i][j];
            }
        }
        input.back() = b.size % 2 ? 1 : -1;
        auto net_output = net.evaluate_const(input);

        std::array<u8, 7> indices{};
        u8 num_valid_indices = 0;
        for (u8 i = 0; i < 7; ++i)
            if (b.data[i].height < 6)
                indices[num_valid_indices++] = i;
        std::default_random_engine rng;
        static i32 m = 1;
        rng.seed(clock() * m++);
        u8 res = std::uniform_int_distribution<u8>{0, num_valid_indices}(rng);

        std::shuffle(indices.begin(), indices.end(), rng);
        for (u8 i = 0; i < net_output.size(); ++i) {
            if (b.data[res].height == 6 && b.data[indices[i]].height < 6) {
                res = indices[i];
            } else if (net_output[indices[i]] > net_output[res] &&
                       b.data[indices[i]].height < 6) {
                res = indices[i];
            }
        }
        return res;
    }
};
}