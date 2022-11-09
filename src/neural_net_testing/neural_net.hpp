#pragma once

#include "../include.hpp"
#include "../board.hpp"
#include "layer_array.hpp"
#include "weight_array.hpp"
#include "optional_structures.hpp"

namespace gya {
template<bool USE_BACKPROP, bool LABELED_DATA, class T, class F1, class F2, u64... sizes>
struct neural_net {
    layer_array<T, sizes...> m_values;
    layer_array<T, sizes...> m_biases;
    weight_array<T, sizes...> m_weights;

    optional_layer_array<USE_BACKPROP, T, sizes...> m_bias_derivatives_acc;
    optional_weight_array<USE_BACKPROP, T, sizes...> m_weight_derivatives_acc;

    optional_layer_array<USE_BACKPROP, T, sizes...> m_bias_derivatives_win;
    optional_weight_array<USE_BACKPROP, T, sizes...> m_weight_derivatives_win;

    optional_layer_array<USE_BACKPROP && !LABELED_DATA, T, sizes...> m_bias_derivatives_loss;
    optional_weight_array<USE_BACKPROP && !LABELED_DATA, T, sizes...> m_weight_derivatives_loss;

    F1 m_activation_function;
    F2 m_activation_derivative;

    template<class F1_, class F2_>
    neural_net(F1_ &&f, F2_ &&derivative) :
            m_activation_function{std::forward<F1_>(f)},
            m_activation_derivative{std::forward<F2_>(derivative)} {}

    auto &operator=(neural_net const &other) {
        m_weights.data = other.m_weights.data;
        m_values.data = other.m_values.data;
        m_biases.data = other.m_biases.data;
        return *this;
    }

    void fill_randomly() {
        *this = {m_activation_function, m_activation_derivative};
        update_randomly(0.1);
    }

    void update_randomly(T amount = 0.1) {
        thread_local std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<T> dist{-amount, amount};
        for (auto &v: m_weights.data)
            v += dist(rng);
        for (auto &v: m_biases.data)
            v += dist(rng);
    }

    static auto compute_cost(std::span<T> output, std::span<T> correct_output) {
        assert(output.size() == correct_output.size());
        T sum = 0;
        for (u64 i = 0; i < output.size(); ++i) {
            sum += (output[i] - correct_output[i]) * (output[i] - correct_output[i]);
        }
        return sum;
    }

    auto compute_derivatives_unlabeled(std::span<T> choice) {
        static_assert(USE_BACKPROP);
        static_assert(!LABELED_DATA);
        thread_local layer_array<T, sizes...> local_bias_derivatives_win;
        thread_local weight_array<T, sizes...> local_weight_derivatives_win;
        thread_local layer_array<T, sizes...> local_bias_derivatives_loss;
        thread_local weight_array<T, sizes...> local_weight_derivatives_loss;

        std::span output{m_values.back()};

        assert(choice.size() == output.size());

        const u64 num_layers = size();
        for (u64 node = 0; node < output.size(); ++node) {
            local_bias_derivatives_win.back()[node] =
                    m_activation_derivative(output[node]) * (output[node] - choice[node]);
            local_bias_derivatives_loss.back()[node] =
                    m_activation_derivative(output[node]) * (output[node] + choice[node]);
        }
        for (u64 layer = num_layers - 2; layer > 0; --layer) {
            for (u64 node = 0; node < m_values[layer].size(); ++node) {
                T sum_win = 0;
                T sum_loss = 0;
                for (u64 next_node = 0; next_node < m_values[layer + 1].size(); ++next_node) {
                    const T weight_derivative_win =
                            m_values[layer][node] * local_bias_derivatives_win[layer + 1][next_node];
                    sum_win += weight_derivative_win;
                    local_weight_derivatives_win[layer][node][next_node] = weight_derivative_win;
                    const T weight_derivative_loss =
                            m_values[layer][node] * local_bias_derivatives_loss[layer + 1][next_node];
                    sum_loss += weight_derivative_loss;
                    local_weight_derivatives_loss[layer][node][next_node] = weight_derivative_loss;
                }
                const T bias_derivative_win = m_activation_derivative(m_values[layer][node]) * sum_win;
                local_bias_derivatives_win[layer][node] = bias_derivative_win;
                const T bias_derivative_loss = m_activation_derivative(m_values[layer][node]) * sum_loss;
                local_bias_derivatives_loss[layer][node] = bias_derivative_loss;
            }
        }
        for (u64 node = 0; node < m_values[0].size(); ++node) {
            for (u64 next_node = 0; next_node < m_values[1].size(); ++next_node) {
                const T weight_derivative_win = m_values[0][node] * local_bias_derivatives_win[1][next_node];
                local_weight_derivatives_win[0][node][next_node] = weight_derivative_win;
                const T weight_derivative_loss = m_values[0][node] * local_bias_derivatives_loss[1][next_node];
                local_weight_derivatives_loss[0][node][next_node] = weight_derivative_loss;
            }
        }
        for (u64 i = 0; i < m_bias_derivatives_win.m_layer_array.data.size(); ++i)
            m_bias_derivatives_win.m_layer_array.data[i] += local_bias_derivatives_win.data[i];
        for (u64 i = 0; i < m_bias_derivatives_loss.m_layer_array.data.size(); ++i)
            m_bias_derivatives_loss.m_layer_array.data[i] += local_bias_derivatives_loss.data[i];
        for (u64 i = 0; i < m_weight_derivatives_win.m_weight_array.data.size(); ++i)
            m_weight_derivatives_win.m_weight_array.data[i] += local_weight_derivatives_win.data[i];
        for (u64 i = 0; i < m_weight_derivatives_loss.m_weight_array.data.size(); ++i)
            m_weight_derivatives_loss.m_weight_array.data[i] += local_weight_derivatives_loss.data[i];
    }

    void compute_derivatives_based_on_result(bool won) {
        static_assert(USE_BACKPROP);
        static_assert(!LABELED_DATA);
        auto &weights_derivatives = won ? m_weight_derivatives_win : m_weight_derivatives_loss;
        auto &bias_derivatives = won ? m_bias_derivatives_win : m_bias_derivatives_loss;
        for (u64 i = 0; i < weights_derivatives.m_weight_array.data.size(); ++i) {
            m_weight_derivatives_acc.m_weight_array.data[i] += weights_derivatives.m_weight_array.data[i];
        }
        for (u64 i = 0; i < bias_derivatives.m_layer_array.data.size(); ++i) {
            m_bias_derivatives_acc.m_layer_array.data[i] += bias_derivatives.m_layer_array.data[i];
        }
        clear_derivatives();
    }

    void clear_derivatives() {
        m_weight_derivatives_win.fill(0);
        m_weight_derivatives_loss.fill(0);
        m_bias_derivatives_win.fill(0);
        m_bias_derivatives_loss.fill(0);
    }

    auto compute_derivatives_labeled(std::span<T> correct_output) {
        static_assert(USE_BACKPROP);
        static_assert(LABELED_DATA);
        std::span output{m_values.back()};
        const u64 num_layers = size();
        for (u64 node = 0; node < output.size(); ++node) {
            m_bias_derivatives_win.back()[node] =
                    m_activation_derivative(output[node]) * (output[node] - correct_output[node]);
        }
        for (u64 layer = num_layers - 2; layer > 0; --layer) {
            for (u64 node = 0; node < m_values[layer].size(); ++node) {
                T sum = 0;
                for (u64 next_node = 0; next_node < m_values[layer + 1].size(); ++next_node) {
                    const T weight_derivative =
                            m_values[layer][node] * m_bias_derivatives_win[layer + 1][next_node];
                    sum += weight_derivative;
                    m_weight_derivatives_win[layer][node][next_node] = weight_derivative;
                }
                const T bias_derivative = m_activation_derivative(m_values[layer][node]) * sum;
                m_bias_derivatives_win[layer][node] = bias_derivative;
            }
        }
        for (u64 node = 0; node < m_values[0].size(); ++node) {
            for (u64 next_node = 0; next_node < m_values[1].size(); ++next_node) {
                const T weight_derivative = m_values[0][node] * m_bias_derivatives_win[1][next_node];
                m_weight_derivatives_win[0][node][next_node] = weight_derivative;
            }
        }
        for (u64 i = 0; i < m_bias_derivatives_acc.data.size(); ++i)
            m_bias_derivatives_acc.data[i] += m_weight_derivatives_win.data[i];
        for (u64 i = 0; i < m_weight_derivatives_acc.data.size(); ++i)
            m_weight_derivatives_acc.data[i] += m_weight_derivatives_win.data[i];
    }

    auto apply_derivatives(T learning_rate = 0.01) {
        static_assert(USE_BACKPROP);
        const u64 num_layers = size();
        for (u64 layer = num_layers; layer-- > 0;) {
            for (u64 node = 0; node < m_values[layer].size(); ++node) {
                m_biases[layer][node] += m_bias_derivatives_acc[layer][node] * -learning_rate;
                m_bias_derivatives_acc[layer][node] = 0;
                for (u64 next_node = 0; next_node < m_values[layer + 1].size(); ++next_node) {
                    m_weights[layer][node][next_node] +=
                            m_weight_derivatives_acc[layer][node][next_node] * -learning_rate;
                    m_weight_derivatives_acc[layer][node][next_node] = 0;
                }
            }
        }
        clear_derivatives();
    }

    auto train(std::span<T> input, std::span<T> correct_output, T learning_rate = 0.01) {
        static_assert(USE_BACKPROP);
        static_assert(LABELED_DATA);
        evaluate(input);
        compute_derivatives_labeled(correct_output);
        apply_derivatives(learning_rate);
    }

    [[nodiscard]] auto evaluate_impl(std::span<T> input, layer_array<T, sizes...> &values) const {
        std::span<T> input_layer{values.front()}, output_layer{values.back()};
        std::copy(input.begin(), input.end(), input_layer.begin());
        for (u64 layer = 1; layer < values.size(); ++layer) {
            for (u64 node = 0; node < values[layer].size(); ++node) {
                T sum = m_biases[layer][node];
                for (u64 prev_node = 0; prev_node < values[layer - 1].size(); ++prev_node) {
                    sum += m_weights[layer - 1][prev_node][node] * values[layer - 1][prev_node];
                }
                values[layer][node] = m_activation_function(sum);
            }
        }
        return output_layer;
    }

    auto evaluate(std::span<T> inp) {
        return evaluate_impl(inp, m_values);
    }

    [[nodiscard]] auto evaluate_const(std::span<T> inp) const {
        layer_array<T, sizes...> values;
        auto output = evaluate_impl(inp, values);
        std::array<T, (sizes, ...)> out;
        std::copy(output.begin(), output.end(), out.begin());
        return out;
    }

    constexpr void swap(neural_net const &x) {
        *this = x;
    }

    [[nodiscard]] constexpr u64 size() const {
        return sizeof...(sizes);
    }

    [[nodiscard]] bool operator!=(neural_net const &other) const {
        return m_weights.data != other.m_weights.data || m_biases.data != other.m_biases.data;
    }

    [[nodiscard]] bool operator==(neural_net const &other) const {
        return m_weights.data == other.m_weights.data && m_biases.data == other.m_biases.data;
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << std::setprecision(10000);
        for (T i: m_weights.data)
            oss << i << ' ';
        for (T i: m_biases.data)
            oss << i << ' ';
        return oss.str();
    }

    void from_string(std::string_view s) {
        std::istringstream iss{std::string(s)};
        for (auto &i: m_weights.data)
            iss >> i;
        for (auto &i: m_biases.data)
            iss >> i;
    }
};
}

namespace std {
template<bool b1, bool b2, class T, class F1, class F2, u64... sizes>
void swap(gya::neural_net<b1, b2, T, F1, F2, sizes...> &t1, gya::neural_net<b1, b2, T, F1, F2, sizes...> &t2) {
    auto temp = std::move(t1);
    t1 = std::move(t2);
    t2 = std::move(temp);
}
}
