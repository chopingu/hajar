#pragma once

#include "include.hpp"

namespace heuristic {
    struct Abias {
        int num_moves;
        int n;

        Abias(int num_moves, int n) : num_moves(num_moves), n(n) {}

        f64 evaluate_board(gya::board const &b, i32 steps_left) const {
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
                auto evaluation = evaluate_board(b.play_copy(move), steps_left - 1) * b.turn();
                if (b.play_copy(move).n_vertical_count(n, b.turn()) > b.n_vertical_count(n, b.turn()))
                    evaluation += 1e6 * b.turn();
                if (evaluation > best_eval)
                    best_eval = evaluation;
            }
            return best_eval * b.turn();
        }

        u8 operator()(gya::board const &b) const {
            f64 best_eval = -std::numeric_limits<f64>::max();
            u8 best_move = 0;
            auto actions = b.get_actions();
            lmj::random_shuffle(actions);


            for (u8 move: actions) {
                auto evaluation = evaluate_board(b.play_copy(move), num_moves - 1) * b.turn();
                if (evaluation > best_eval) {
                    best_eval = evaluation;
                    best_move = move;
                }
            }

            return best_move;
        }
    };
}
