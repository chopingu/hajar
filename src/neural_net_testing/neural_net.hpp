#include <random>
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

        void fill_randomly() {
            std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<T> dist{-1.0f, 1.0f};
            for (auto &v: m_weights.data)
                v = dist(rng);
            for (auto &v: m_biases.data)
                v = dist(rng);
        }

        auto calculate_deltas(std::span<T> correct_output) const {
            std::span output{m_values.back()};
            assert(correct_output.size() == output.size());
            layer_array < T, sizes...> desired_values;
            desired_values.back() = correct_output;
            layer_array < T, sizes...> bias_deltas;
            weight_array < T, sizes...> weight_deltas;

            for (u64 layer = size(); layer > 0; --layer) {

            }

            for (u64 layer = size(); layer > 0; --layer) {
                for (u64 node = 0; node < m_values[layer]; ++node) {
                    for (u64 prev_node = 0; prev_node < m_values[layer - 1]; ++prev_node) {
                        const T prev_val = m_values[layer - 1][prev_node];
                        const T curr_val = m_values[layer][node];
                        const T corr_val = desired_values[layer][node];
                        const T delta = prev_val * activation_derivative(curr_val) * 2 * (curr_val - corr_val);
                        weight_deltas[layer][prev_node][node] = delta;
                    }
                }
            }
            return std::tuple{bias_deltas, weight_deltas};
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
            const T cost_before = compute_cost(m_values.back(), correct_output);
            calculate_deltas(correct_output);

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
            layer_array < T, sizes...> values;
            auto output = evaluate_impl(inp, values);
            std::array<T, (sizes, ...)> out;
            std::copy(output.begin(), output.end(), out.begin());
            return out;
        }

        constexpr u64 size() const {
            return sizeof...(sizes);
        }
    };
}