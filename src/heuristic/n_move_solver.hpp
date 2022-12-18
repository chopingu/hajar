#pragma once

#include "../include.hpp"
#include "eval_result.hpp"

namespace heuristic {
struct n_move_solver {
    i32 m_depth = 5;
    bool multi_thread = false;

    [[nodiscard]] constexpr eval_result evaluate_board(gya::board const &board, i32 depth = -1) const {
        if (depth == -1)
            depth = m_depth - 1;
        if (board.has_won().player_1_won())
            return board.turn() == gya::board::PLAYER_ONE ? WINNING_MOVE : LOSING_MOVE;
        if (board.has_won().player_2_won())
            return board.turn() == gya::board::PLAYER_TWO ? WINNING_MOVE : LOSING_MOVE;
        if (board.has_won().is_tie())
            return TIE_MOVE;
        if (depth == 0)
            return NEUTRAL_MOVE;

        eval_result best_eval = LOSING_MOVE;
        for (u8 move: board.get_actions()) {
            eval_result eval = evaluate_board(board.play_copy(move), depth - 1).incremented();
            if (eval > best_eval)
                best_eval = eval;
            if (eval.m_depth_until_over == 1 || eval.is_winning() == true)
                break;
        }
        return best_eval;
    }

    [[nodiscard]] u8 operator()(gya::board const &board) const {
        u8 best_move = gya::BOARD_WIDTH;
        eval_result best_eval = LOSING_MOVE;
        auto actions = board.get_actions();
        // put indices closer to the middle first
        std::sort(std::begin(actions), std::end(actions), [](u8 lhs, u8 rhs) {
            return std::abs(lhs - gya::BOARD_WIDTH / 2) <
                   std::abs(rhs - gya::BOARD_WIDTH / 2);
        });

        if (!multi_thread || m_depth < 5) {
            for (u8 move: actions) {
                const auto eval = evaluate_board(board.play_copy(move)).incremented();
                if (best_move == gya::BOARD_WIDTH || eval > best_eval)
                    best_move = move, best_eval = eval;
            }
        } else {
            std::array<bool, 7> used{};
            std::array<eval_result, 7> evaluations{};
            lmj::static_vector<std::future<void>, 7> futures;
            for (u8 move: actions) {
                futures.emplace_back(std::async(std::launch::async,
                                                [&, move] {
                                                    evaluations[move] = evaluate_board(
                                                            board.play_copy(move)).incremented();
                                                    used[move] = true;
                                                }
                ));
            }
            for (auto &&i: futures) i.get();
            for (auto move: actions) {
                if (used[move]) {
                    if (evaluations[move] > best_eval) {
                        best_eval = evaluations[move];
                        best_move = move;
                    }
                }
            }
        }
        return best_move;
    }
};
}
