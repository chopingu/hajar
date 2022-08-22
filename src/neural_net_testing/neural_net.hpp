#include <random>
#include "board.hpp"
#include "layer_array.hpp"
#include "weight_array.hpp"
#include "defines.hpp"

namespace gya {
    template<class T, u64... sizes>
    struct neural_net {
        layer_array<T, sizes...> m_values;
        layer_array<T, sizes...> m_biases;
        weight_array<T, sizes...> m_weights;
        std::span<T> input{m_values.front()}, output{m_values.back()};

        neural_net() {
            fill_randomly();
        }

        void fill_randomly() {
            std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<T> dist{0.0f, 1.0f};
            for (auto &v: m_weights.data)
                v = dist(rng);
            for (auto &v: m_biases.data)
                v = dist(rng);
        }

        static float activation(float x) {
            return std::clamp(0.0f, 1.0f, x * 0.2f + 0.5f); // 867.696ns to infer
            return 1.0f / (1.0f + std::exp(-x)); // 454727ns to infer
        }

        [[nodiscard]] std::span<T> evaluate(std::span<T> inp) {
            // 1 2 3 4 5 6 7
            // 8 9 . . .
            std::copy(inp.begin(), inp.end(), input.begin());

            for (u64 i = 0; i + 1 < m_values.size(); ++i) {
                for (u64 j = 0; j < m_values[i].size(); ++j) {
                    for (u64 k = 0; k < m_values[i + 1].size(); ++k) {
                        m_values[i + 1][k] = activation(m_values[i][j] * m_weights[i][j][k] + m_biases[i + 1][k]);
                    }
                }
            }

            return output;
        }
    };
}