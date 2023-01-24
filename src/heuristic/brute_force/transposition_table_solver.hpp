#pragma once

#include "../../include.hpp"
#include "eval_result.hpp"

namespace heuristic {
struct transposition_table_solver {
    struct compressed_board_hasher {
        constexpr usize operator()(gya::compressed_board const &b) const noexcept {
            return lmj::compute_hash(std::bit_cast<const char *>(&b), sizeof(b));
        };
    };

    // constexpr static compressed_board_hasher hasher{};

    i32 m_depth = 5;
    lmj::hash_table<gya::compressed_board, heuristic::eval_result, compressed_board_hasher> m_ttable{};

    [[nodiscard]] eval_result evaluate_board(gya::board const &board) { return evaluate_board(board, m_depth - 1); }

    [[nodiscard]] eval_result evaluate_board(gya::board const &board, i32 depth) {
        if (board.has_won().is_game_over()) {
            if (board.has_won().player_1_won())
                return board.turn() == gya::board::PLAYER_ONE ? WINNING_MOVE : LOSING_MOVE;
            if (board.has_won().player_2_won())
                return board.turn() == gya::board::PLAYER_TWO ? WINNING_MOVE : LOSING_MOVE;
            if (board.has_won().is_tie()) return TIE_MOVE;
        }
        if (depth == 0) return NEUTRAL_MOVE;

        if (auto iter = m_ttable.find(board); iter != m_ttable.end())
            return iter->second;

        eval_result best_eval = LOSING_MOVE;
        auto actions = board.get_actions();
        std::sort(std::begin(actions), std::end(actions), [](u8 lhs, u8 rhs) {
            return std::abs(lhs - gya::BOARD_WIDTH / 2) < std::abs(rhs - gya::BOARD_WIDTH / 2);
        });
        for (u8 move: actions) {
            eval_result eval = evaluate_board(board.play_copy(move), depth - 1).incremented();
            if (eval > best_eval) best_eval = eval;
            if (eval.is_winning()) break;
        }

        if (best_eval.is_game_over()) m_ttable.emplace(board, best_eval);
        return best_eval;
    }

    [[nodiscard]] u8 operator()(gya::board const &board) {
        //for (auto &[b, eval]: m_ttable)
            //if (gya::compressed_board::decompress(b).num_played_moves() < board.num_played_moves()) m_ttable.erase(b);
        u8 best_move = gya::BOARD_WIDTH;
        eval_result best_eval = LOSING_MOVE;
        auto actions = board.get_actions();
        // put indices closer to the middle first
        std::sort(std::begin(actions), std::end(actions), [](u8 lhs, u8 rhs) {
            return std::abs(lhs - gya::BOARD_WIDTH / 2) < std::abs(rhs - gya::BOARD_WIDTH / 2);
        });

        for (u8 move: actions) {
            const auto eval = evaluate_board(board.play_copy(move)).incremented();
            if (best_move == gya::BOARD_WIDTH || eval > best_eval) best_move = move, best_eval = eval;
        }

        return best_move;
    }
};
} // namespace heuristic
