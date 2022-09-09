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
            std::uniform_real_distribution<T> dist{-0.5f, 0.5f};
            for (auto &v: m_weights.data)
                v = dist(rng);
            for (auto &v: m_biases.data)
                v = dist(rng);
        }

        layer_array<T, sizes...> calculate_deltas(std::span<T> correct_output) const {
            std::span<T> input{m_values.front()}, output{m_values.back()};
            layer_array < T, sizes...> deltas;
            for (u64 i = 0; i < output.size(); ++i) {
                deltas[i] = output[i] - correct_output[i];
            }
            for (u64 i = size(); i-- > 1;) {

            }
        }

        auto back_propagate(std::span<T> correct_output) {
            const auto deltas = calculate_deltas(correct_output);
            std::span<T> input{m_values.front()}, output{m_values.back()};

        }

        [[nodiscard]] auto evaluate_impl(std::span<T> inp, layer_array<T, sizes...>& values) const {
            std::span<T> input{values.front()}, output{values.back()};
            std::copy(inp.begin(), inp.end(), input.begin());

            for (u64 i = 0; i + 1 < values.size(); ++i) {
                for (u64 j = 0; j < values[i].size(); ++j) {
                    values[i][j] = activation_function(values[i][j]);
                    for (u64 k = 0; k < values[i + 1].size(); ++k) {
                        values[i + 1][k] += values[i][j] * m_weights[i][j][k] + m_biases[i + 1][k];
                    }
                }
            }

            return output;
        }

        [[nodiscard]] auto evaluate(std::span<T> inp) {
            return evaluate_impl(inp, m_values);
        }

        [[nodiscard]] auto evaluate_const(std::span<T> inp) const {
            layer_array < T, sizes...> values;
            auto output = evaluate_impl(values);
            return std::array(output.begin(), output.end());
        }

        constexpr u64 size() const {
            return sizeof...(sizes);
        }
    };
}