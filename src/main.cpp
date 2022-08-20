#include "board.hpp"
#include "tester.hpp"
#include <bits/stdc++.h>

int main() {
    {
        // test logic
        auto a = gya::board{
                gya::board_column{std::array<uint8_t, 6>{2, 2, 2, 1}, 4},
                gya::board_column{std::array<uint8_t, 6>{2, 2, 1}, 3},
                gya::board_column{std::array<uint8_t, 6>{2, 1}, 2},
                gya::board_column{std::array<uint8_t, 6>{1}, 1},
                gya::board_column{std::array<uint8_t, 6>{}, 0},
                gya::board_column{std::array<uint8_t, 6>{}, 0},
                gya::board_column{std::array<uint8_t, 6>{}, 0},
        };
        auto b = gya::board{
                gya::board_column{std::array<uint8_t, 6>{1}, 1},
                gya::board_column{std::array<uint8_t, 6>{2, 1}, 2},
                gya::board_column{std::array<uint8_t, 6>{2, 2, 1}, 3},
                gya::board_column{std::array<uint8_t, 6>{2, 2, 2, 1}, 4},
                gya::board_column{std::array<uint8_t, 6>{}, 0},
                gya::board_column{std::array<uint8_t, 6>{}, 0},
                gya::board_column{std::array<uint8_t, 6>{}, 0},
        };
        if (!a.has_won_test()) {
            std::cout << "game logic broken, should be win" << a.to_string() << std::endl;
        }
        if (!b.has_won_test()) {
            std::cout << "game logic broken, should be win" << b.to_string() << std::endl;
        }

        gya::random_player p1, p2;
        gya::board c = gya::test_game(p1, p2);
        while (c.has_won() != -1) {
            c = gya::test_game(p1, p2);
        }
        if (c.has_won_test() != -1) {
            std::cout << "game logic is broken, should be tie:\n" << c.to_string() << std::endl;
            return 0;
        } else {
            std::cout << "game logic is ok" << std::endl;
        }
    }

    {
        // test performance
        gya::random_player p1, p2;
        for (int i = 0; i < 10; i++) {
            gya::board c = gya::test_game(p1, p2);
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < (1 << 18); ++j)
                c = gya::test_game(p1, p2);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto time = (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)) / static_cast<double>(1 << 18);
            if (time > std::chrono::nanoseconds(4000)) { //
                std::cout << "performance is too slow: " << time.count() << " ns" << std::endl;
                return 0;
            }
        }
        std::cout << "performance is ok" << std::endl;
    }

    {
        // test randomness
        gya::random_player p;
        std::unordered_map<uint64_t, uint64_t> prev_vals;
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
