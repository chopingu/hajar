#pragma once

#include "../include.hpp"

namespace heuristic {
struct n_move_solver_simple {
    int num_moves = 5;

    f64 evaluate_board(gya::board const &b, i32 steps_left, i32 recursion_depth = 0) const {
        if (gya::game_result result = b.has_won(); result.is_game_over()) {
            if (result.is_tie()) {
                return -1e5 * b.turn();
            } else if (result.player_1_won()) {
                return 1e9;
            } else {
                return -1e9;
            }
        }

        if (steps_left == 0) return 0;

        const auto actions = b.get_actions();

        f64 best_eval = 0;

        for (auto move: actions) {
            const auto evaluation = evaluate_board(b.play_copy(move), steps_left - 1) * b.turn();
            if (evaluation > best_eval)
                best_eval = evaluation;
        }

        return best_eval;
    }

    u8 operator()(gya::board const &b) const {
        f64 best_eval = -1e9;
        u8 best_move = 0;
        auto actions = b.get_actions();
        lmj::random_shuffle(actions);

        for (u8 move: actions) {
            const auto evaluation = evaluate_board(b.play_copy(move), num_moves - 1) * b.turn();
            if (evaluation > best_eval) {
                best_eval = evaluation;
                best_move = move;
            }
        }

        return best_move;
    }
};
}