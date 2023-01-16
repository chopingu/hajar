#pragma once

#include "../include.hpp"
#include "eval_result.hpp"

namespace heuristic {
struct n_move_solver {
    i32 m_depth = 5;
    bool multi_thread = false;

    [[nodiscard]] eval_result evaluate_board(gya::board const &board) const {
        return evaluate_board(board, m_depth - 1);
    }

    [[nodiscard]] eval_result evaluate_board(gya::board const &board, i32 depth) const {
        if (board.has_won().player_1_won()) // someone already won
            return board.turn() == gya::board::PLAYER_ONE ? WINNING_MOVE : LOSING_MOVE;
        if (board.has_won().player_2_won()) // someone already won
            return board.turn() == gya::board::PLAYER_TWO ? WINNING_MOVE : LOSING_MOVE;
        if (board.has_won().is_tie()) // game is tied
            return TIE_MOVE;
        if (depth == 0) // if we're out of depth and the game isn't over, we say it's neutral
            return NEUTRAL_MOVE;

        eval_result best_eval = LOSING_MOVE;
        for (u8 move: board.get_actions()) {
            // look at the state after playing the current move, recurse
            // .incremented() flips the winning/losing state and increments the number of moves
            // until the winning move if one is found
            eval_result eval = evaluate_board(board.play_copy(move), depth - 1).incremented();

            // an eval is said to be better than another eval if it's either a better result (eg winning vs tied)
            // or if it's temporally better (winning faster or losing later)
            if (eval > best_eval)
                best_eval = eval;

            // if the move we just played leads to the game being over (either we win or tie immediately),
            // or if we find a winning sequence of moves we may stop evaluating
            // this may lead to not finding a shorter sequence of winning moves, but this is unlikely
            // may be worth investigating
            if (eval.is_winning()) break;
        }
        return best_eval;
    }

    [[nodiscard]] u8 operator()(gya::board const &board) const {
        u8 best_move = gya::BOARD_WIDTH;
        eval_result best_eval = LOSING_MOVE;
        auto actions = board.get_actions();
        // put indices closer to the middle first
        std::sort(std::begin(actions), std::end(actions), [](u8 lhs, u8 rhs) {
            return std::abs(lhs - gya::BOARD_WIDTH / 2) < std::abs(rhs - gya::BOARD_WIDTH / 2);
        });

        if (!multi_thread || m_depth < 5) {
            for (u8 move: actions) {
                const auto eval = evaluate_board(board.play_copy(move)).incremented();
                if (best_move == gya::BOARD_WIDTH || eval > best_eval)
                    best_move = move, best_eval = eval;
            }
        } else { // multithreaded vv ^^ not multithreaded, look above for readability
            std::array<bool, 7> used{};
            std::array<eval_result, 7> evaluations{};
            lmj::static_vector<std::future<void>, 7> futures;
            for (u8 move: actions) {
                futures.emplace_back(std::async(std::launch::async, [&, move] {
                    evaluations[move] = evaluate_board(board.play_copy(move)).incremented();
                    used[move] = true;
                }));
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
} // namespace heuristic
