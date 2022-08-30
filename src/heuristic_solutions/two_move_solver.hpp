#include "../defines.hpp"
#include "../board.hpp"
#include "../tester.hpp"

namespace gya {

    struct two_move_solver {
        gya::random_player rp{};

        u8 operator()(gya::board const &b) {
            i8 turn = b.size % 2 ? 1 : -1;
            std::array<u8, 7> candidates{};
            u8 num_candidates = 0;
            for (u8 i = 0; i < 7; ++i) {
                gya::board copy = b;
                if (copy.data[i].height == 6)
                    continue;
                else
                    candidates[num_candidates++] = i;
                copy.play(i, turn);
                if (copy.has_won().state == turn)
                    return i;
            }
            turn = turn == 1 ? -1 : 1;
            for (u8 i = 0; i < 7; ++i) {
                gya::board copy = b;
                if (copy.data[i].height == 6)
                    continue;
                copy.play(i, turn);
                if (copy.has_won().state == turn)
                    return i;
            }
            turn = turn == 1 ? -1 : 1;
            for (int i = 0; i < 16; ++i) {
                auto move = rp(b);
                gya::board copy = b;
                copy.play(move, turn);
                if (copy.data[move].height < 6)
                    copy.play(move, turn == 1 ? -1 : 1);
                if (copy.has_won_test().state != (turn == 1 ? -1 : 1))
                    return move;
            }
            return rp(b);
        }

    };
}
