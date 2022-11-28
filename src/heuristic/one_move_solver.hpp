#pragma once

#include "../include.hpp"

namespace heuristic {
struct one_move_solver {
    gya::random_player m_random_player{};

    u8 operator()(gya::board const &b) {
        i8 turn = b.size % 2 ? 1 : -1;
        for (u8 i = 0; i < 7; ++i) {
            gya::board copy = b;
            if (copy.data[i].height == 6)
                continue;
            copy.play(i, turn);
            if (copy.has_won().state == turn)
                return i;
        }
        return m_random_player(b);
    }
};
}
