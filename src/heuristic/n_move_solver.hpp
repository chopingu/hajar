#pragma once

#include "../include.hpp"

namespace heuristic {
template<bool MULTI_THREAD = false>
struct n_move_solver {
    struct eval_result {
        bool m_winning: 1;
        bool m_losing: 1;
        i8 m_depth_until_over: 6 = 0; 

        constexpr eval_result() : m_winning{}, m_losing{}, m_depth_until_over{} {}

        constexpr eval_result(bool winning, bool losing, i8 depth = 0) : m_winning{winning}, m_losing{losing}, m_depth_until_over{depth} {}

        constexpr eval_result incremented() const {
            if (m_winning | m_losing) {
                return eval_result{!m_winning, !m_losing, static_cast<i8>(m_depth_until_over + 1)};
            } else {
                return *this;
            }
        }

        constexpr bool operator>(eval_result other) const {
            if (m_winning && !other.m_winning)
                return true;
            if (!m_losing && other.m_losing)
                return true;
            if (m_winning && other.m_winning)
                return m_depth_until_over < other.m_depth_until_over;
            if (m_losing && other.m_losing)
                return m_depth_until_over > other.m_depth_until_over;
            return false;
        }

        constexpr bool operator<(eval_result other) const {
            return other > *this;
        }

        operator const char*() const {
            if (m_winning)
                return "WINNING";
            if (m_losing) 
                return "LOSING";
            return "NEUTRAL";
        }
    };
    static_assert(sizeof(eval_result) == 1);

    static constexpr eval_result WINNING_MOVE{true, false};
    static constexpr eval_result LOSING_MOVE{false, true};
    static constexpr eval_result NEUTRAL_MOVE{};

    i32 m_depth = 5;


    [[nodiscard]] eval_result evaluate_board(gya::board const& board, i32 remaining_moves = -1) const {
        if (remaining_moves == -1) remaining_moves = m_depth;

        if (board.has_won().player_1_won()) {
            return board.turn() == gya::board::PLAYER_ONE ? WINNING_MOVE : LOSING_MOVE;
        }
        if (board.has_won().player_2_won()) {
            return board.turn() == gya::board::PLAYER_TWO ? WINNING_MOVE : LOSING_MOVE;
        }
        if (board.has_won().is_tie() || remaining_moves == 0) {
            return NEUTRAL_MOVE;
        } else {
            eval_result best_eval = LOSING_MOVE;        
            for (u8 move: board.get_actions()) {
                eval_result eval = evaluate_board(board.play_copy(move), remaining_moves - 1).incremented();
                if (eval > best_eval) 
                    best_eval = eval;
            }
            return best_eval;
        }
    }

    [[nodiscard]] u8 operator()(gya::board const& board) const {
        u8 best_move = -1;
        eval_result best_move_eval = LOSING_MOVE;
        for (u8 move: board.get_actions()) {
            const auto eval = evaluate_board(board.play_copy(move)).incremented();
            if (best_move == static_cast<u8>(-1) || eval > best_move_eval)
                best_move = move, best_move_eval = eval;
        }
        return best_move;
    }
};
}
