#include "defines.hpp"
#include "board.hpp"
#include "tester.hpp"
#include <iostream>
#include "neural_net_testing/neural_net_player.hpp"

int main() {
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
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < (1 << 20); ++j)
            volatile gya::board c = gya::test_game(p1, p2);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)) / static_cast<double>(1 << 20);
        using std::chrono_literals::operator ""ns;
        std::cout << "random player:\n";
        std::cout << "avg: " << time.count() << "ns" << std::endl;

    }

    {
        // test neural net performance
        gya::random_player p1;
        gya::neural_net_player p2;

        using dur_type = std::chrono::duration<double, std::ratio<1, 1000000000>>;
        dur_type avg{0};
        dur_type min{1e9};
        dur_type max{0};
        constexpr int iters = 1 << 14;
        double avg_num_moves = 0;
        for (int i = 0; i < iters; ++i) {
            const auto t1 = std::chrono::high_resolution_clock::now();
            int move_count = 0;
            for (int j = 0; j < 16; ++j) {
                gya::board b = gya::test_game(p1, p2);
                move_count += b.size / 2;
            }
            const auto t2 = std::chrono::high_resolution_clock::now();
            const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1) / 16.0;
            avg_num_moves += move_count / 16.0;
            min = std::min(min, time);
            max = std::max(max, time);
            avg += time;
        }
        std::cout << "neural net:\n";
        std::cout << "avg_num_moves: " << avg_num_moves / iters << std::endl;
        std::cout << "avg: " << (avg / iters).count() << "ns" << std::endl;
        std::cout << "min: " << (min).count() << "ns" << std::endl;
        std::cout << "max: " << (max).count() << "ns" << std::endl;
    }

    std::cout << "tests passed" << std::endl;
    std::cin.get();
}
