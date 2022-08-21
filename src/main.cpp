#include "defines.hpp"
#include "board.hpp"
#include "tester.hpp"
#include <iostream>

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
        // test performance
        gya::random_player p1, p2;
        for (int i = 0; i < 10; i++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < (1 << 20); ++j)
                volatile gya::board c = gya::test_game(p1, p2);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto time = (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)) / static_cast<double>(1 << 20);

            using std::chrono_literals::operator ""ns;
            if (time > 4000ns) {
                std::cout << "performance is too slow: " << time.count() << " ns" << std::endl;
                return 0;
            }
        }
        std::cout << "performance is ok" << std::endl;
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

    std::cout << "tests passed" << std::endl;
    std::cin.get();
}
