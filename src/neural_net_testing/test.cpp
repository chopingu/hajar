#include "defines.hpp"
#include "neural_net_player.hpp"
#include "heuristic_solutions/one_move_solver.hpp"
#include "heuristic_solutions/two_move_solver.hpp"
#include <bits/stdc++.h>

namespace gya {
    void randomly_update_neural_net_player(gya::neural_net_player &p, f32 change_rate) {
        u64 num_changes = std::random_device{}() % (p.size() / 8);
        std::default_random_engine rng{std::random_device{}()};
        auto rand_f = [&](f32 low, f32 high) { return std::uniform_real_distribution<f32>{low, high}(rng); };
        auto rand_i = [&](u64 low, u64 high) { return std::uniform_int_distribution<u64>{low, high}(rng); };
        for (u64 i = 0; i < num_changes; ++i) {
            if (rand_f(0, 1) < 0.5f) {
                auto &item = p.net.m_weights.data[rand_i(0, p.net.m_weights.data.size() - 1)];
                item = std::clamp(item + rand_f(-change_rate, change_rate), -1.0f, 1.0f);
            } else {
                auto &item = p.net.m_biases.data[rand_i(0, p.net.m_biases.data.size() - 1)];
                item = std::clamp(item + rand_f(-change_rate, change_rate), -1.0f, 1.0f);
            }
        }
    }

    gya::board get_starting_pos(i32 num_random_moves) {
        gya::board starting_pos;
        gya::random_player p;
        for (i32 i = 0; i < num_random_moves; ++i)
            starting_pos.play(p(starting_pos), i % 2 == 0 ? 1 : -1);
        return starting_pos;
    }
}

int main() {
    constexpr auto num_players = 50;

    constexpr auto surviving_portion = 0.9;

//    gya::two_move_solver untrained_player;
    gya::neural_net_player untrained_player;

    std::vector<gya::neural_net_player> players(num_players);

    for (auto &player: players)
        player.net.fill_randomly();

    for (int iter = 0;; ++iter) {
        try {
            {
//                lmj::timer t;
                int wins_1 = 0, wins_2 = 0;
                int ties_1 = 0, ties_2 = 0;
                int losses_1 = 0, losses_2 = 0;
                int num_games = 47;
                for (int i = 0; i < num_games; ++i) {
                    for (auto &player: players) {
                        auto r1 = gya::test_game(player, untrained_player, gya::get_starting_pos(0)).has_won();
                        auto r2 = gya::test_game(untrained_player, player, gya::get_starting_pos(0)).has_won();
                        wins_1 += r1.player_1_won();
                        wins_2 += r2.player_2_won();
                        ties_1 += r1.is_tie();
                        ties_2 += r2.is_tie();
                        losses_1 += r1.player_2_won();
                        losses_2 += r2.player_1_won();
                    }
                }

                std::cout << "trained goes first:\n"
                          << "wins: " << wins_1 * 1e2 / num_games / num_players << "%\n"
                          << "ties: " << ties_1 * 1e2 / num_games / num_players << "%\n"
                          << "losses: " << losses_1 * 1e2 / num_games / num_players << "%\n";
                std::cout << "untrained goes first:\n"
                          << "wins: " << wins_2 * 1e2 / num_games / num_players << "%\n"
                          << "ties: " << ties_2 * 1e2 / num_games / num_players << "%\n"
                          << "losses: " << losses_2 * 1e2 / num_games / num_players << "%\n";
                std::cout << "combined\n"
                          << "wins: " << (wins_1 + wins_2) / 2.0 * 1e2 / num_games / num_players << "%\n"
                          << "ties: " << (ties_1 + ties_2) / 2.0 * 1e2 / num_games / num_players << "%\n"
                          << "losses: " << (losses_1 + losses_2) / 2.0 * 1e2 / num_games / num_players << "%\n";
                std::cout << std::endl;
            }
            for (int iter2 = 0; iter2 < 24; ++iter2) {
//                lmj::timer t;
                std::array<std::pair<i32, u64>, num_players> num_wins{};
                for (int i = 0; i < 1; ++i) {
                    auto last_gen = players;

                    for (i32 j = num_players * surviving_portion + 1; j < num_players; ++j)
                        gya::randomly_update_neural_net_player(players[j],
                                                               std::clamp(0.2f, 0.8f, 1.0f / (iter * 0.01f + 1.0f)));
//                        players[j].net.fill_randomly();
                    std::vector<std::thread> threads;
                    for (u64 j = 0; j < num_players; ++j) {
                        std::atomic<i32> wins = 0;
                        threads.emplace_back([j, &players, &last_gen, &wins] {
                            i32 local_wins = 0;
                            for (u64 k = 0; k < num_players; ++k) {
                                auto &p1 = players[j];
                                 auto &p2 = last_gen[k];
                                const auto board = gya::get_starting_pos(rand() % 5 * 2);
                                const auto result_1 = gya::test_game(p1, p2, board).has_won();
                                const auto result_2 = gya::test_game(p2, p1, board).has_won();
                                if (result_1.player_1_won()) ++local_wins;
                                if (result_1.player_2_won()) --local_wins;

                                if (result_2.player_1_won()) --local_wins;
                                if (result_2.player_2_won()) ++local_wins;
                            }
                            wins += local_wins;
                        });
                        num_wins[j] = {wins, j};
                    }
                    for (auto &j: threads)
                        j.join();
                    std::sort(num_wins.rbegin(), num_wins.rend());

                    std::function<void(i32, i32)> recursive_swap = [&](i32 a, i32 b) {
                        if (a != b) {
                            std::swap(players[a], players[b]);
                            recursive_swap(a, num_wins[b].second);
                        }
                    };

                    for (i32 j = 0; j < num_players; ++j)
                        recursive_swap(j, num_wins[j].second);

                    for (u64 j = 0; j / surviving_portion < num_players; ++j)
                        for (u64 k = 0; k < 1 / surviving_portion; ++k)
                            players[surviving_portion * num_players + k] = players[j];
                }
                std::cerr << std::flush;
            }
        } catch (std::exception const &e) {
            std::cerr << "ERROR\n";
            std::cerr << e.what() << std::endl;
//            std::cin.get();
        }
    }
}
