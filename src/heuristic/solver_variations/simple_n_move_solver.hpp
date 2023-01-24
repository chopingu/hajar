#pragma once

#include "../../include.hpp"

namespace heuristic {
struct n_move_solver_simple {
    u32 m_num_moves;

    n_move_solver_simple(u32 num_moves) : m_num_moves(num_moves) {}

    f64 evaluate_board(gya::board const &b, u32 steps_left) const {
        if (gya::game_result result = b.has_won(); result.is_game_over()) {
            if (result.is_tie()) {
                return -1e5;
            } else if (result.player_1_won()) {
                return (b.turn() == gya::board::PLAYER_ONE ? 1 : -1) * 1e9;
            } else {
                return (b.turn() == gya::board::PLAYER_TWO ? 1 : -1) * 1e9;
            }
        }

        if (steps_left == 0) return 0;

        f64 best_eval = -1e10;
        for (auto move: b.get_actions()) {
            const auto evaluation = evaluate_board(b.play_copy(move), steps_left - 1) * -1;
            if (evaluation > best_eval)
                best_eval = evaluation;
        }
        return best_eval;
    }

    u8 operator()(gya::board const &b) const {
        f64 best_eval = -std::numeric_limits<f64>::max();
        u8 best_move = 0;
        auto actions = b.get_actions();
        // lmj::random_shuffle(actions);
        // for (u8 move: actions) {
        //     if (b.play_copy(move).n_vertical_count(m_n, -1) > b.n_vertical_count(m_n, -1))
        //         return move;
        // }

        for (u8 move: actions) {
            const auto evaluation = evaluate_board(b.play_copy(move), m_num_moves - 1) * b.turn();
            if (evaluation > best_eval) {
                best_eval = evaluation;
                best_move = move;
            }
        }

        return best_move;
    }
};
} // namespace heuristic
