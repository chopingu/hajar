#include "defines.hpp"
#include "neural_net.hpp"
#include "board.hpp"

namespace gya {
    struct neural_net_player {
        neural_net<f32, 42, 7> net;

        [[nodiscard]] u8 operator()(gya::board const &b) {
            std::array<f32, 42> input{};
            for (u64 i = 0; i < 6; ++i) {
                for (u64 j = 0; j < 7; ++j) {
                    input[i * 7 + j] = b.data[i][j];
                }
            }
            auto net_output = net.evaluate(input);

            std::array<u8, 7> indices;
            std::iota(indices.begin(), indices.end(), 0);
            std::default_random_engine rng;
            static i32 m = 1;
            rng.seed(clock() * m++);
            u8 res = std::uniform_int_distribution<u8>{0, 7}(rng);

            std::shuffle(indices.begin(), indices.end(), rng);
            for (u8 i = 0; i < net_output.size(); ++i) {
                if (b.data[res].height == 6 && b.data[indices[i]].height < 6) {
                    res = indices[i];
                } else if (net_output[indices[i]] > net_output[res] && b.data[indices[i]].height < 6) {
                    res = indices[i];
                }
            }
            return res;
        }
    };
}