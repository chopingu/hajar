#pragma once

#include <algorithm>
#include <random>
#include <future>
#include "../defines.hpp"
#include "../board.hpp"
#include "../tester.hpp"
#include "../../lib/lmj/src/lmj_include_all.hpp"

namespace heuristic {
struct n_move_solver {
    constexpr static auto MULTI_THREAD = true;

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
        constexpr std::array<int, 7> biases = {0, 1, 2, 3, 2, 1, 0};
//        constexpr std::array<int, 7> biases = {};

        std::array<f64, 7> scores{};
        lmj::static_vector<f64, 7> move_scores;
        const auto actions = b.get_actions();
        const auto eval_move = [=, &scores, &biases, &move_scores](u8 move) {
            const auto evaluation = evaluate_board(b.play_copy(move), steps_left - 1, recursion_depth + 1)
                                    * 0.75 * b.turn() + biases[move];
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
                if (scores[move] >= 0.75 * 1e9) {
                    return scores[move] * b.turn();
                }
            }
        }
        if (move_scores.size() >= 3) {
            std::partial_sort(move_scores.begin(), move_scores.begin() + 3, move_scores.end(), std::greater{});
            f64 score = move_scores[0];
            score += 1e-8 * move_scores[1] + 1e-9 * move_scores[2];
            return score * b.turn();
        } else {
            std::sort(move_scores.begin(), move_scores.end(), std::greater{});
            f64 score = move_scores[0];
            if (move_scores.size() == 2) score += move_scores[1] * 1e-8;
            return score * b.turn();
        }
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
            moves.emplace_back(evaluate_board(b.play_copy(move), num_moves - 1) * b.turn(), move);
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
