#pragma once

#include <random>
#include <sstream>
#include <iomanip>
#include <memory>
#include "board.hpp"
#include "layer_array.hpp"
#include "weight_array.hpp"
#include "defines.hpp"

namespace gya {
    template<class T, class F1, class F2, u64... sizes>
    struct neural_net {
        layer_array<T, sizes...> m_values;
        layer_array<T, sizes...> m_biases;
        weight_array<T, sizes...> m_weights;
        F1 activation_function;
        F2 activation_derivative;

        neural_net(F1 f, F2 derivative) : activation_function{f}, activation_derivative{derivative} {}

        auto &operator=(neural_net const &other) {
            m_weights.data = other.m_weights.data;
            m_values.data = other.m_values.data;
            m_biases.data = other.m_biases.data;
            return *this;
        }

        void fill_randomly() {
            thread_local std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<T> dist{-0.1f, 0.1f};
            for (auto &v: m_weights.data)
                v = dist(rng);
            for (auto &v: m_biases.data)
                v = dist(rng);
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

        auto train(std::span<T> input, std::span<T> correct_output, T learning_rate = 0.01) {
            evaluate(input);
            // const T cost_before = compute_cost(m_values.back(), correct_output);
            layer_array<T, sizes...> bias_derivatives;
            thread_local auto weight_derivatives = std::make_unique<weight_array<T, sizes...>>();
            std::span output{m_values.back()};
            const u64 num_layers = size();
            for (u64 node = 0; node < output.size(); ++node) {
                bias_derivatives.back()[node] =
                        activation_derivative(output[node]) * (output[node] - correct_output[node]);
            }
            for (u64 layer = num_layers - 2; layer > 0; --layer) {
                for (u64 node = 0; node < m_values[layer].size(); ++node) {
                    T sum = 0;
                    for (u64 next_node = 0; next_node < m_values[layer + 1].size(); ++next_node) {
                        const T weight_derivative = m_values[layer][node] * bias_derivatives[layer + 1][next_node];
                        sum += weight_derivative;
                        (*weight_derivatives)[layer][node][next_node] = weight_derivative;
                    }
                    const T bias_derivative = activation_derivative(m_values[layer][node]) * sum;
                    bias_derivatives[layer][node] = bias_derivative;
                }
            }
            for (u64 node = 0; node < m_values[0].size(); ++node) {
                for (u64 next_node = 0; next_node < m_values[1].size(); ++next_node) {
                    const T weight_derivative = m_values[0][node] * bias_derivatives[1][next_node];
                    (*weight_derivatives)[0][node][next_node] = weight_derivative;
                }
            }

            for (u64 layer = 0; layer < num_layers; ++layer) {
                for (u64 node = 0; node < m_values[layer].size(); ++node) {
                    m_biases[layer][node] += bias_derivatives[layer][node] * -learning_rate;
                    for (u64 next_node = 0; next_node < m_values[layer + 1].size(); ++next_node) {
                        m_weights[layer][node][next_node] +=
                                (*weight_derivatives)[layer][node][next_node] * -learning_rate;
                    }
                }
            }
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
                    values[layer][node] = activation_function(sum);
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

        constexpr u64 size() const {
            return sizeof...(sizes);
        }

        bool operator!=(neural_net const& other) const {
            return m_weights.data != other.m_weights.data || m_biases.data != other.m_biases.data;
        }

        bool operator==(neural_net const& other) const {
            return !(*this == other);
        }

        std::string to_string() const {
            std::ostringstream oss;
            oss << std::setprecision(10000);
            for (T i: m_weights.data)
                oss << i << ' ';
            for (T i: m_biases.data)
                oss << i << ' ';
            return oss.str();
        }

        void from_string(std::string_view s) {
            std::istringstream iss((std::string(s)));
            for (auto &i: m_weights.data)
                iss >> i;
            for (auto &i: m_biases.data)
                iss >> i;
        }
    };
}

namespace std {
    template<class T, class F1, class F2, u64... sizes>
    void swap(gya::neural_net<T, F1, F2, sizes...> &t1, gya::neural_net<T, F1, F2, sizes...> &t2) {
        auto temp = std::move(t1);
        t1 = std::move(t2);
        t2 = std::move(temp);
    }
}