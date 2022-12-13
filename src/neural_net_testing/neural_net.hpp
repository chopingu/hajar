#pragma once

#include "../include.hpp"
#include "../board.hpp"
#include "layer_array.hpp"
#include "weight_array.hpp"
#include "optional_structures.hpp"

namespace gya {
template<bool USE_BACKPROP, bool LABELED_DATA, class T, class F1, class F2, usize... sizes>
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

    // for passing around derivatives
    using derivative_pair_t = std::pair<weight_array<T, sizes...>, layer_array<T, sizes...>>;

    F1 m_activation_function;
    F2 m_activation_derivative;

    template<class F1_, class F2_>
    constexpr neural_net(F1_ &&f, F2_ &&derivative) :
            m_activation_function{std::forward<F1_>(f)},
            m_activation_derivative{std::forward<F2_>(derivative)} {}

    constexpr auto &operator=(neural_net const &other) {
        m_weights.m_data = other.m_weights.m_data;
        m_values.m_data = other.m_values.m_data;
        m_biases.m_data = other.m_biases.m_data;
        return *this;
    }

    constexpr neural_net(neural_net const &other) : m_activation_function{other.m_activation_function},
                                                    m_activation_derivative{other.m_activation_derivative} {
        *this = other;
    }

    void fill_randomly() {
        *this = {m_activation_function, m_activation_derivative};
        update_randomly(0.1);
    }

    void update_randomly(T amount = 0.1) {
        thread_local std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<T> dist{-amount, amount};
        for (auto &v: m_weights.m_data)
            v += dist(rng);
        for (auto &v: m_biases.m_data)
            v += dist(rng);
    }

    static auto compute_cost(std::span<T> output, std::span<T> correct_output) {
        assert(output.size() == correct_output.size());
        T sum = 0;
        for (usize i = 0; i < output.size(); ++i) {
            sum += (output[i] - correct_output[i]) * (output[i] - correct_output[i]);
        }
        return sum;
    }

    [[deprecated]] auto compute_derivatives_unlabeled(std::span<T> choice) {
        static_assert(USE_BACKPROP);
        static_assert(!LABELED_DATA);
        thread_local layer_array<T, sizes...> local_bias_derivatives_win;
        thread_local weight_array<T, sizes...> local_weight_derivatives_win;
        thread_local layer_array<T, sizes...> local_bias_derivatives_loss;
        thread_local weight_array<T, sizes...> local_weight_derivatives_loss;

        std::span output{m_values.back()};

        assert(choice.size() == output.size());

        const usize num_layers = size();
        for (usize node = 0; node < output.size(); ++node) {
            local_bias_derivatives_win.back()[node] =
                    m_activation_derivative(output[node]) * (output[node] - choice[node]);
            local_bias_derivatives_loss.back()[node] =
                    m_activation_derivative(output[node]) * (output[node] + choice[node]);
        }
        for (usize layer = num_layers - 2; layer > 0; --layer) {
            for (usize node = 0; node < m_values[layer].size(); ++node) {
                T sum_win = 0;
                T sum_loss = 0;
                for (usize next_node = 0; next_node < m_values[layer + 1].size(); ++next_node) {
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
        for (usize node = 0; node < m_values[0].size(); ++node) {
            for (usize next_node = 0; next_node < m_values[1].size(); ++next_node) {
                const T weight_derivative_win = m_values[0][node] * local_bias_derivatives_win[1][next_node];
                local_weight_derivatives_win[0][node][next_node] = weight_derivative_win;
                const T weight_derivative_loss = m_values[0][node] * local_bias_derivatives_loss[1][next_node];
                local_weight_derivatives_loss[0][node][next_node] = weight_derivative_loss;
            }
        }
        for (usize i = 0; i < m_bias_derivatives_win.m_layer_array.m_data.size(); ++i)
            m_bias_derivatives_win.m_layer_array.m_data[i] += local_bias_derivatives_win.m_data[i];
        for (usize i = 0; i < m_bias_derivatives_loss.m_layer_array.m_data.size(); ++i)
            m_bias_derivatives_loss.m_layer_array.m_data[i] += local_bias_derivatives_loss.m_data[i];
        for (usize i = 0; i < m_weight_derivatives_win.m_weight_array.m_data.size(); ++i)
            m_weight_derivatives_win.m_weight_array.m_data[i] += local_weight_derivatives_win.m_data[i];
        for (usize i = 0; i < m_weight_derivatives_loss.m_weight_array.m_data.size(); ++i)
            m_weight_derivatives_loss.m_weight_array.m_data[i] += local_weight_derivatives_loss.m_data[i];
    }

    [[deprecated]] void compute_derivatives_based_on_result(bool won) {
        static_assert(USE_BACKPROP);
        static_assert(!LABELED_DATA);
        auto &weights_derivatives = won ? m_weight_derivatives_win : m_weight_derivatives_loss;
        auto &bias_derivatives = won ? m_bias_derivatives_win : m_bias_derivatives_loss;
        for (usize i = 0; i < weights_derivatives.m_weight_array.m_data.size(); ++i) {
            m_weight_derivatives_acc.m_weight_array.m_data[i] += weights_derivatives.m_weight_array.m_data[i];
        }
        for (usize i = 0; i < bias_derivatives.m_layer_array.m_data.size(); ++i) {
            m_bias_derivatives_acc.m_layer_array.m_data[i] += bias_derivatives.m_layer_array.m_data[i];
        }
        clear_derivatives();
    }

    void clear_derivatives() {
        m_weight_derivatives_win.fill(0);
        m_weight_derivatives_loss.fill(0);
        m_bias_derivatives_win.fill(0);
        m_bias_derivatives_loss.fill(0);
    }

    [[deprecated]] auto compute_derivatives_labeled_impl(std::span<T> correct_output) {
        static_assert(USE_BACKPROP);
        static_assert(LABELED_DATA);
        std::span output{m_values.back()};
        const usize num_layers = size();
        for (usize node = 0; node < output.size(); ++node) {
            m_bias_derivatives_win.back()[node] =
                    m_activation_derivative(output[node]) * (output[node] - correct_output[node]);
        }
        for (usize layer = num_layers - 2; layer > 0; --layer) {
            for (usize node = 0; node < m_values[layer].size(); ++node) {
                T sum = 0;
                for (usize next_node = 0; next_node < m_values[layer + 1].size(); ++next_node) {
                    const T weight_derivative =
                            m_values[layer][node] * m_bias_derivatives_win[layer + 1][next_node];
                    sum += weight_derivative;
                    m_weight_derivatives_win[layer][node][next_node] = weight_derivative;
                }
                const T bias_derivative = m_activation_derivative(m_values[layer][node]) * sum;
                m_bias_derivatives_win[layer][node] = bias_derivative;
            }
        }
        for (usize node = 0; node < m_values[0].size(); ++node) {
            for (usize next_node = 0; next_node < m_values[1].size(); ++next_node) {
                const T weight_derivative = m_values[0][node] * m_bias_derivatives_win[1][next_node];
                m_weight_derivatives_win[0][node][next_node] = weight_derivative;
            }
        }
    }

    [[deprecated]] auto compute_derivatives_labeled(std::span<T> correct_output) {
        compute_derivatives_labeled_impl(correct_output);
        for (usize i = 0; i < m_bias_derivatives_acc.data.size(); ++i)
            m_bias_derivatives_acc.data[i] += m_weight_derivatives_win.data[i];
        for (usize i = 0; i < m_weight_derivatives_acc.data.size(); ++i)
            m_weight_derivatives_acc.data[i] += m_weight_derivatives_win.data[i];
    }

    [[deprecated]] auto apply_derivatives(T learning_rate = 0.01) {
        static_assert(USE_BACKPROP);
        const usize num_layers = size();
        for (usize layer = num_layers; layer-- > 0;) {
            for (usize node = 0; node < m_values[layer].size(); ++node) {
                m_biases[layer][node] += m_bias_derivatives_acc[layer][node] * -learning_rate;
                m_bias_derivatives_acc[layer][node] = 0;
                for (usize next_node = 0; next_node < m_values[layer + 1].size(); ++next_node) {
                    m_weights[layer][node][next_node] +=
                            m_weight_derivatives_acc[layer][node][next_node] * -learning_rate;
                    m_weight_derivatives_acc[layer][node][next_node] = 0;
                }
            }
        }
        clear_derivatives();
    }

    [[deprecated]] auto train(std::span<T> input, std::span<T> correct_output, T learning_rate = 0.01) {
        static_assert(USE_BACKPROP);
        static_assert(LABELED_DATA);
        evaluate(input);
        compute_derivatives_labeled(correct_output);
        apply_derivatives(learning_rate);
    }

    [[nodiscard]] std::span<T> forward_propagate(std::span<T> input, layer_array<T, sizes...> &values) const {
        std::span<T> input_layer{values.front()}, output_layer{values.back()};
        std::copy(input.begin(), input.end(), input_layer.begin());
        for (usize layer = 1; layer < values.size(); ++layer) {
            for (usize node = 0; node < values[layer].size(); ++node) {
                T sum = m_biases[layer][node];
                for (usize prev_node = 0; prev_node < values[layer - 1].size(); ++prev_node) {
                    sum += m_weights[layer - 1][prev_node][node] * values[layer - 1][prev_node];
                }
                values[layer][node] = m_activation_function(sum);
            }
        }
        return output_layer;
    }

    [[nodiscard]] constexpr derivative_pair_t
    backward_propagate(layer_array<T, sizes...> const &node_values,
                       std::span<T> correct_output,
                       bool post_activation_variables) const {
        // result variables
        weight_array<T, sizes...> weight_derivatives{};
        layer_array<T, sizes...> bias_derivatives{};

        const std::span output{node_values.back()};
        const usize num_layers = size();

        // process last layer
        for (usize node = 0; node < output.size(); ++node) {
            bias_derivatives.back()[node] =
                    m_activation_derivative(output[node]) * (output[node] - correct_output[node]);
        }

        // process all layers between last and first layer (exclusive)
        for (usize layer = num_layers - 2; layer > 0; --layer) {
            for (usize node = 0; node < node_values[layer].size(); ++node) {
                T sum = 0;
                for (usize next_node = 0; next_node < node_values[layer + 1].size(); ++next_node) {
                    const T weight_derivative = node_values[layer][node] * bias_derivatives[layer + 1][next_node];
                    sum += weight_derivative;
                    weight_derivatives[layer][node][next_node] = weight_derivative;
                }
                const T bias_derivative = m_activation_derivative(node_values[layer][node]) * sum;
                bias_derivative[layer][node] = bias_derivative;
            }
        }

        // process first layer
        for (usize node = 0; node < node_values[0].size(); ++node) {
            for (usize next_node = 0; next_node < node_values[1].size(); ++next_node) {
                const T weight_derivative = node_values[0][node] * bias_derivatives[1][next_node];
                weight_derivatives[0][node][next_node] = weight_derivative;
            }
        }

        return {weight_derivatives, bias_derivatives};
    }

    void apply_derivatives(derivative_pair_t const &derivatives, T learning_rate) {
        auto &[weight_derivatives, bias_derivatives] = derivatives;
        for (usize i = 0; i < weight_derivatives.size(); ++i) {
            m_weights.data()[i] += weight_derivatives.data()[i];
        }
    }

    [[nodiscard]] std::span<T> evaluate(std::span<T> inp) {
        return forward_propagate(inp, m_values);
    }

    [[nodiscard]] std::array<T, util::get_last(sizes...)> evaluate_const(std::span<T> inp) const {
        layer_array<T, sizes...> values;
        std::span<T> output = forward_propagate(inp, values);
        std::array<T, util::get_last(sizes...)> out_arr{};
        std::copy(output.begin(), output.end(), out_arr.begin());
        return out_arr;
    }

    constexpr void swap(neural_net const &x) {
        *this = x;
    }

    [[nodiscard]] constexpr usize size() const {
        return sizeof...(sizes);
    }

    [[nodiscard]] bool operator!=(neural_net const &other) const {
        return m_weights.m_data != other.m_weights.m_data || m_biases.m_data != other.m_biases.m_data;
    }

    [[nodiscard]] bool operator==(neural_net const &other) const {
        return m_weights.m_data == other.m_weights.m_data && m_biases.m_data == other.m_biases.m_data;
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << std::setprecision(10000);
        for (T i: m_weights.m_data)
            oss << i << ' ';
        for (T i: m_biases.m_data)
            oss << i << ' ';
        return oss.str();
    }

    void from_string(std::string_view s) {
        std::istringstream iss{std::string(s)};
        for (auto &i: m_weights.m_data)
            iss >> i;
        for (auto &i: m_biases.m_data)
            iss >> i;
    }
};
}

namespace std {
template<bool b1, bool b2, class T, class F1, class F2, usize... sizes>
void swap(gya::neural_net<b1, b2, T, F1, F2, sizes...> &t1, gya::neural_net<b1, b2, T, F1, F2, sizes...> &t2) {
    auto temp = std::move(t1);
    t1 = std::move(t2);
    t2 = std::move(temp);
}
}
