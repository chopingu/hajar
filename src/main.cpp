#include "board.hpp"
#include "tester.hpp"
#include <bits/stdc++.h>

int main() {
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
    assert(a.has_won());
    assert(b.has_won());

    // play until tie
    gya::random_player p1, p2;
    gya::board c = gya::test_game(p1, p2);
    while (c.has_won() != -1) {
        std::cout << c.to_string() << std::endl;
        c = gya::test_game(p1, p2);
    }
    std::cout << c.to_string() << std::endl;

    // test performance
    // gya::random_player p1, p2;
    for (int i = 0; i < 10; i++) {
        gya::board c = gya::test_game(p1, p2);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < (1 << 20); ++j)
            c = gya::test_game(p1, p2);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "time: "
                  << static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) /
                     static_cast<double>(1 << 20)
                  << "ns per game" << std::endl;
    }

    std::cin.get();
}
