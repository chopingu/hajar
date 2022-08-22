#include "defines.hpp"
#include "neural_net.hpp"
#include "board.hpp"

namespace gya {
    struct neural_net_player {
        neural_net<float, 42, 80, 64, 7> net;

        [[nodiscard]] u8 operator()(gya::board const &b) {
            std::array<float, 42> input{};
            for (u64 i = 0; i < 6; ++i) {
                for (u64 j = 0; j < 7; ++j) {
                    input[i * 7 + j] = b.data[i][j];
                }
            }
            auto net_output = net.evaluate(input);

            u8 res = 0;
            for (u8 i = 0; i < net_output.size(); ++i) {
                if (b.data[res].height == 6 && b.data[i].height < 6) {
                    res = i;
                } else if (net_output[i] > net_output[res] && b.data[i].height < 6) {
                    res = i;
                }
            }
            return res;
        }
    };
}