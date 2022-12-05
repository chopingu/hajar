#pragma once

#include "../include.hpp"

namespace heuristic {
struct one_move_solver {
    gya::random_player m_random_player{};

    u8 operator()(gya::board const &b) {
        for (u8 move: b.get_actions()) {
            if (b.is_winning_move(move).is_game_over()) {
                return move;
            }
        }
        return m_random_player(b);
    }
};
}
