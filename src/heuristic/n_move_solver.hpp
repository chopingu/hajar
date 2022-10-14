#pragma once

#include <algorithm>
#include <random>
#include "../defines.hpp"
#include "../board.hpp"
#include "../tester.hpp"

namespace heuristic {
struct n_move_solver {
    f32 evaluate_board(gya::board const &b, i32 steps_left) const {
        if (gya::game_result result = b.has_won(); result.is_game_over()) {
            if (result.is_tie()) {
                return 0;
            } else if (result.player_1_won()) {
                return 1e6;
            } else {
                return -1e6;
            }
        }
        if (steps_left == 0) return 0;
        f32 best_score = -1e9;
        for (u8 move: b.get_actions()) {
            f32 temp = evaluate_board(b.play_copy(move), steps_left - 1) * 0.75 * b.turn();
            if (temp > best_score) {
                best_score = temp;
            }
        }
        return best_score * b.turn();
    }

    u8 operator()(gya::board const &b) const {
        f32 best_score = -1e9;
        u8 best_move = 0;
        auto actions = b.get_actions();
        std::shuffle(actions.begin(), actions.end(), std::default_random_engine{static_cast<u32>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        for (u8 move: actions) {
            f32 temp = evaluate_board(b.play_copy(move), 5) * b.turn();
            if (temp > best_score) {
                best_score = temp;
                best_move = move;
            }
        }
        return best_move;
    }
};
}
