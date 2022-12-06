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

        constexpr eval_result(bool winning, bool losing, i8 depth = 0) : m_winning{winning}, m_losing{losing},
                                                                         m_depth_until_over{depth} {}

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

        constexpr operator const char *() const {
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

    [[nodiscard]] eval_result evaluate_board(gya::board const &board, i32 remaining_moves = -1) const {
        if (remaining_moves == -1) remaining_moves = m_depth - 1;

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
                if (eval.m_depth_until_over == 1 || eval.m_winning == true)
                    break;
            }
            return best_eval;
        }
    }

    [[nodiscard]] u8 operator()(gya::board const &board) const {
        u8 best_move = gya::BOARD_WIDTH;
        eval_result best_eval = LOSING_MOVE;
        auto actions = board.get_actions();
        // put indices closer to the middle first
        std::sort(std::begin(actions), std::end(actions), [](u8 lhs, u8 rhs) {
            return std::abs(lhs - gya::BOARD_WIDTH / 2) < std::abs(rhs - gya::BOARD_WIDTH / 2);
        });

        if (!MULTI_THREAD || m_depth < 5) {
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
