#include "include.hpp"

#include "neural_net_testing/neural_net_player.hpp"
#include "neural_net_testing/neural_net_player_deep.hpp"

#include "heuristic/mcts.hpp"
#include "heuristic/one_move_solver.hpp"
#include "heuristic/two_move_solver.hpp"
#include "heuristic/n_move_solver.hpp"
#include "heuristic/solver_variations/A.hpp"
#include "heuristic/solver_variations/Abias.hpp"

#include "pinguml/tensor.hpp"
#include "pinguml/activation.hpp"
#include "pinguml/cost.hpp"

int main() {
    /*
    auto print = [](std::string_view s) {
        for (auto c: s) {
            putchar(c);
        }
    };

    while (true) {
        gya::board b;
        int turn = 1;
        heuristic::Abias s(5, 3);
        while (!b.has_won().is_game_over()) {
            if (turn ^= 1) {
                print(b.to_string());
//                std::cout << (int) heuristic::n_move_solver{}.evaluate_board(b, 5) * b.turn() << '\n';
                u8 move;
                {
                    lmj::timer t{false};
                    move = s(b);
                    printf("%fs\n", t.elapsed());
                }
                b.play(move);
            } else {
                print(b.to_string());
//                std::cout << (int) heuristic::n_move_solver{}.evaluate_board(b, 5) * b.turn() << '\n';
                int move;
                std::cin >> move;
                b.play(move - 1);
            }
        }
        print(b.to_string());
        if (b.has_won().is_tie()) {
            std::puts("tie!");
        } else if (b.has_won().player_1_won()) {
            std::puts("you won!");
        } else {
            std::puts("you lost!");
        }
    }
    */

    {
        // tests
        {
            // test logic
            auto a = gya::board::from_string(
                    "| | | | | | | |\n"
                    "| | | | | | | |\n"
                    "|X| | | | | | |\n"
                    "|O|X| | | | | |\n"
                    "|O|O|X| | | | |\n"
                    "|O|O|O|X| | | |\n"
                    "|1|2|3|4|5|6|7|\n"
            );
            auto b = gya::board::from_string(
                    "| | | | | | | |\n"
                    "| | | | | | | |\n"
                    "| | | |X| | | |\n"
                    "| | |X|O| | | |\n"
                    "| |X|O|O| | | |\n"
                    "|X|O|O|O| | | |\n"
                    "|1|2|3|4|5|6|7|\n"
            );

            if (gya::board::from_string(a.to_string()) != a ||
                gya::board::from_string(b.to_string()) != b) {
                std::cout << "string conversion is broken" << std::endl;
                return 0;
            }

            if (!a.has_won().player_1_won() || !a.has_won_test().player_1_won()) {
                std::cout << "game logic broken, should be win for X\n" << a.to_string() << std::endl;
                return 0;
            }
            if (!b.has_won().player_1_won() || !b.has_won_test().player_1_won()) {
                std::cout << "game logic broken, should be win for X\n" << b.to_string() << std::endl;
                return 0;
            }

            gya::random_player p1, p2;
            gya::board c = gya::test_game(p1, p2);
            while (!c.has_won().is_tie()) {
                c = gya::test_game(p1, p2);
            }

            if (!c.has_won_test().is_tie()) {
                std::cout << "game logic is broken, should be tie:\n" << c.to_string() << std::endl;
                return 0;
            }
            std::cout << "game logic is ok" << std::endl;
        }

        {
            // test randomness
            gya::random_player p;
            std::unordered_map<u64, u64> prev_vals;
            for (int i = 0; i < (1 << 20); ++i) {
                if (++prev_vals[p.get_num()] > 2) {
                    std::cout << "randomness is shite" << std::endl;
                    return 0;
                }
            }
            std::cout << "randomness is ok" << std::endl;
        }

        {
            // test performance
            gya::random_player p1, p2;
            constexpr auto NUM_GAMES = 1 << 10;
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < NUM_GAMES; ++j)
                [[maybe_unused]] volatile gya::board const c = gya::test_game(p1, p2);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto time =
                    (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)) / static_cast<double>(NUM_GAMES);
            using std::chrono_literals::operator ""ns;
            std::cout << "random player:\n";
            std::cout << "avg: " << time.count() << "ns" << std::endl;

        }

        {
            // test neural m_net runtime  performance
            gya::random_player p1;
            gya::neural_net_player p2;

            constexpr auto NANOSECONDS_PER_SECOND = 1'000'000'000;

            using dur_type = std::chrono::duration<double, std::ratio<1, NANOSECONDS_PER_SECOND>>;
            dur_type avg{0};
            dur_type min{1e9};
            dur_type max{0};
            constexpr int iters = 1 << 8;
            double avg_num_moves = 0;

            constexpr auto NUM_GAMES = 16;

            for (usize i = 0; i < iters; ++i) {
                const auto t1 = std::chrono::high_resolution_clock::now();
                usize move_count = 0;
                for (usize j = 0; j < NUM_GAMES; ++j) {
                    gya::board const b = gya::test_game(p1, p2);
                    move_count += b.size / 2;
                }
                const auto t2 = std::chrono::high_resolution_clock::now();
                const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1) / 16.0;
                avg_num_moves += move_count / static_cast<double>(NUM_GAMES);
                min = std::min(min, time);
                max = std::max(max, time);
                avg += time;
            }
            std::cout << "neural m_net:\n";
            std::cout << "avg_num_moves: " << avg_num_moves / iters << std::endl;
            std::cout << "avg: " << (avg / iters).count() << "ns" << std::endl;
            std::cout << "min: " << (min).count() << "ns" << std::endl;
            std::cout << "max: " << (max).count() << "ns" << std::endl;
        }

        {
            // test heuristic solvers
            {
                heuristic::n_move_solver<false> const s{4};
                gya::random_player p;
                constexpr auto NUM_ITERS = 1000;
                constexpr auto NUM_GAMES = NUM_ITERS * 2;
                int wins = 0;
                for (int i = 0; i < NUM_ITERS; ++i) {
                    wins += gya::test_game(s, p).has_won().player_1_won();
                    wins += gya::test_game(p, s).has_won().player_2_won();
                }
                std::cout << "4 move solver win-rate against random player: " << wins * 1e2 / NUM_GAMES << "%"
                          << std::endl;
            }
            {
                heuristic::n_move_solver<false> const s{5}, p{2};
                constexpr auto NUM_ITERS = 1000;
                constexpr auto NUM_GAMES = NUM_ITERS * 2;
                int wins = 0;
                for (int i = 0; i < NUM_ITERS; ++i) {
                    wins += gya::test_game(s, p).has_won().player_1_won();
                    wins += gya::test_game(p, s).has_won().player_2_won();
                }
                std::cout << "4 move solver win-rate against 2 move solver: " << wins * 1e2 / NUM_GAMES << "%"
                          << std::endl;
            }
            {
                heuristic::n_move_solver<false> const s{4};
                heuristic::two_move_solver p;
                constexpr auto NUM_ITERS = 1000;
                constexpr auto NUM_GAMES = NUM_ITERS * 2;
                int wins = 0;
                for (usize i = 0; i < NUM_ITERS; ++i) {
                    wins += gya::test_game(s, p).has_won().player_1_won();
                    wins += gya::test_game(p, s).has_won().player_2_won();
                }
                std::cout << "4 move solver win-rate against two_move_solver: " << wins * 1e2 / NUM_GAMES << "%"
                          << std::endl;
            }
            {
                heuristic::one_move_solver s;
                gya::board board_1 = gya::board::from_string(
                        "| | | | | | | |\n"
                        "| | | | | | | |\n"
                        "| | | | | | | |\n"
                        "|O|X| | | | | |\n"
                        "|O|O|X| | | | |\n"
                        "|X|O|O|X| | | |\n"
                        "|1|2|3|4|5|6|7|\n"
                );
                board_1.play(s(board_1), 1);
                assert(board_1.has_won_test().player_1_won());

                board_1 = gya::board::from_string(
                        "| | | | | | | |\n"
                        "| | | | | | | |\n"
                        "| | | | | | | |\n"
                        "| | | | | | | |\n"
                        "| | | |O| |O| |\n"
                        "| |O|X|X| |X| |\n"
                        "|1|2|3|4|5|6|7|\n"
                );
                board_1.play(s(board_1), 1);
                assert(board_1.has_won_test().player_1_won());
            }
            {
                heuristic::one_move_solver p1, p2;
                constexpr auto NUM_GAMES = 1 << 10;
                auto const t1 = std::chrono::high_resolution_clock::now();
                for (usize j = 0; j < (1 << 10); ++j)
                    [[maybe_unused]] volatile gya::board const c = gya::test_game(p1, p2);
                auto const t2 = std::chrono::high_resolution_clock::now();
                auto const time =
                        (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)) / static_cast<double>(NUM_GAMES);
                using std::chrono_literals::operator ""ns;
                std::cout << "one_move_solver:\n";
                std::cout << "avg: " << time.count() / NUM_GAMES << "us" << std::endl;
            }
            {
                heuristic::two_move_solver p1, p2;
                constexpr auto NUM_GAMES = 1 << 10;
                auto const t1 = std::chrono::high_resolution_clock::now();
                for (usize j = 0; j < NUM_GAMES; ++j)
                    [[maybe_unused]] volatile gya::board const c = gya::test_game(p1, p2);
                auto const t2 = std::chrono::high_resolution_clock::now();
                auto const time =
                        (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)) / static_cast<double>(NUM_GAMES);
                using std::chrono_literals::operator ""ns;
                std::cout << "two_move_solver:\n";
                std::cout << "avg: " << time.count() / NUM_GAMES << "us" << std::endl;
            }
            {
                heuristic::n_move_solver<false> const p1{5}, p2{5};
                constexpr auto NUM_GAMES = 1 << 10;
                auto const t1 = std::chrono::high_resolution_clock::now();
                for (usize j = 0; j < NUM_GAMES; ++j)
                    [[maybe_unused]] volatile gya::board const c = gya::test_game(p1, p2);
                auto const t2 = std::chrono::high_resolution_clock::now();
                auto const time = (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)) / static_cast<double>(NUM_GAMES);
                using std::chrono_literals::operator ""ns;
                std::cout << "n_move_solver (5 moves):\n";
                std::cout << "avg: " << time.count() / NUM_GAMES << "us" << std::endl;
            }
        }

        std::cout << "tests passed" << std::endl;
//        std::cin.get();
    }
}
