#pragma once

#include "board.hpp"
#include "defines.hpp"

namespace gya {
template<class player1_t, class player2_t>
gya::board test_game(player1_t &&player1, player2_t &&player2, gya::board b = {}) {
    i32 turn = 0;
    while (!b.has_won_test().is_game_over()) {
        turn ^= 1;
        if (turn) {
            b.play(player1(b), 1);
        } else {
            b.play(player2(b), -1);
        }
    }
    return b;
}
}
