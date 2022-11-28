#pragma once

#include "../include.hpp"

namespace heuristic {
template<bool MULTI_THREAD = false>
struct n_move_solver {

    int m_num_moves = 5;

    using transpo_table_t = lmj::hash_table<gya::compressed_board, i8>;

    i16 evaluate_board(gya::board const &b, i32 steps_left, i32 recursion_depth = 0) const {
        if (gya::game_result result = b.has_won(); result.is_game_over()) {
            if (result.is_tie()) {
                return static_cast<i16>(-127 * b.turn());
            } else if (result.player_1_won()) {
                return 32767;
            } else {
                return -32767;
            }
        }

        if (steps_left == 0) return 0;
        constexpr std::array<int, 7> biases = {0, 1, 2, 3, 2, 1, 0};
//        constexpr std::array<int, 7> biases = {};

        std::array<i16, 7> scores{};
        lmj::static_vector<i16, 7> move_scores;
        const auto actions = b.get_actions();
        const auto eval_move = [=, this, &scores, &biases, &move_scores](u8 move) {
            const auto evaluation = static_cast<i16>(
                    (evaluate_board(b.play_copy(move), steps_left - 1, recursion_depth + 1)
                     - 10) * b.turn() + biases[move]);
            scores[move] = evaluation;
            move_scores.push_back(evaluation);
        };
        if (recursion_depth < 2 && actions.size() > 3 && MULTI_THREAD) {
            lmj::static_vector<std::future<void>, 7> futures;
            for (u8 move: actions) {
                futures.push_back(std::async(std::launch::async, eval_move, move));
            }
            for (auto &&future: futures) {
                future.get();
            }
        } else {
            for (u8 move: actions) {
                eval_move(move);
                if (scores[move] >= 32757) {
                    return static_cast<i16>(scores[move] * b.turn());
                }
            }
        }
        return static_cast<i16>(*std::max_element(std::begin(move_scores), std::end(move_scores)) * b.turn());
    }

    u8 operator()(gya::board const &b) const {
        f64 best_score = -1e9;
        u8 best_move = 0;
        auto actions = b.get_actions();
//        std::shuffle(actions.begin(), actions.end(), std::default_random_engine{
//                static_cast<u32>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        lmj::random_shuffle(actions);
        lmj::static_vector<std::pair<f64, u8>, 7> moves;

        for (u8 move: actions) {
            moves.emplace_back(evaluate_board(b.play_copy(move), m_num_moves - 1) * b.turn(), move);
        }

        for (auto [score, move]: moves) {
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }

        return best_move;
    }
};
}
