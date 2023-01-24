#pragma once

#include "../../include.hpp"

namespace heuristic {

struct two_move_solver {
    gya::random_player m_random_player{};

    u8 operator()(gya::board const &b) {
        i8 turn = b.turn();
        for (u8 i = 0; i < 7; ++i) {
            gya::board copy = b;
            if (copy.data[i].height == 6)
                continue;
            copy.play(i);
            if ((turn == 1 && copy.has_won().player_1_won()) || (turn == -1 && copy.has_won().player_2_won()))
                return i;
        }
        turn = -turn;
        for (u8 i = 0; i < 7; ++i) {
            gya::board copy = b;
            if (copy.data[i].height == 6)
                continue;
            copy.play(i, turn);
            if ((turn == 1 && copy.has_won().player_1_won()) || (turn == -1 && copy.has_won().player_2_won()))
                return i;
        }
        turn = -turn;
        for (int i = 0; i < 16; ++i) {
            auto move = m_random_player(b);
            gya::board copy = b;
            copy.play(move, turn);
            if (copy.data[move].height < 6)
                copy.play(move, -turn);
            if ((turn == -1 && copy.has_won().player_1_won()) || (turn == 1 && copy.has_won().player_2_won()))
                return move;
        }
        return m_random_player(b);
    }
};
} // namespace heuristic
