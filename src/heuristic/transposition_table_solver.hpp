#pragma once

#include "../include.hpp"
#include "eval_result.hpp"

namespace heuristic {
struct transposition_table_solver {
    constexpr static usize hash_bytes(std::span<const char> bytes) {
        usize result = sizeof(usize) == 8 ? 14695981039346656037ULL : 2166136261U;

        for (auto i: bytes) {
            result ^= i;
            result *= sizeof(usize) == 8 ? 1099511628211ULL : 16777619U;
        }
        return result;
    }

    struct compressed_board_hasher {
        usize operator()(gya::compressed_board const& b) const noexcept {
            return hash_bytes(std::span(reinterpret_cast<const char*>(&b), sizeof(b)));
        };
    };

    constexpr static compressed_board_hasher hasher{};

    i32 m_depth = 5;
    std::unordered_map<gya::compressed_board, heuristic::eval_result, compressed_board_hasher> m_ttable{};

    [[nodiscard]] eval_result evaluate_board(gya::board const &board, i32 depth = -1) {
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

        if (auto iter = m_ttable.find(board); iter != m_ttable.end())
            return iter->second;

        eval_result best_eval = LOSING_MOVE;
        for (u8 move: board.get_actions()) {
            eval_result eval = evaluate_board(board.play_copy(move), depth - 1).incremented();
            if (eval > best_eval)
                best_eval = eval;
            if (eval.m_depth_until_over == 1 || eval.m_winning == true)
                break;
        }

        if (best_eval.is_game_over())
            m_ttable.emplace(board, best_eval);
        return best_eval;
    }

    [[nodiscard]] u8 operator()(gya::board const& board) {
        u8 best_move = gya::BOARD_WIDTH;
        eval_result best_eval = LOSING_MOVE;
        auto actions = board.get_actions();
        // put indices closer to the middle first
        std::sort(std::begin(actions), std::end(actions), [](u8 lhs, u8 rhs) {
            return std::abs(lhs - gya::BOARD_WIDTH / 2) <
                   std::abs(rhs - gya::BOARD_WIDTH / 2);
        });

        for (u8 move : actions) {
            const auto eval = evaluate_board(board.play_copy(move)).incremented();
            if (best_move == gya::BOARD_WIDTH || eval > best_eval)
                best_move = move, best_eval = eval;
        }

        return best_move;
    }
};
}
