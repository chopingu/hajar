#pragma once

#include <chrono>
#include <iostream>
#include "board.hpp"

namespace gya {
    template<class player1_t, class player2_t>
    gya::board test_game(player1_t &player1, player2_t &player2) {
        board b;
        int turn = 0;
        while (!b.has_won_test()) {
//            if (b.has_won() != b.has_won_test()) {
//                std::cout << "Error:\n" << b.to_string() << ' ' << b.has_won() << ' ' << b.has_won_test() << std::endl;
//                exit(-1);
//            }
            turn ^= 1;
            if (turn) {
                b.play(player1(b), 1);
            } else {
                b.play(player2(b), 2);
            }
        }
        return b;
    }

    struct random_player {
        std::uint64_t x = 123456789, y = 362436069, z = 521288629;

        random_player() {
            static std::uint64_t offs = 0;
            std::uint64_t t = ++offs * (std::chrono::system_clock::now().time_since_epoch().count() & 0xffffffff);
            x ^= t;
            y ^= (t >> 32) ^ (t << 32);
            z ^= (t >> 16) ^ (t << 48);

            for (int i = 0; i < 128; ++i)
                get_num();
        }

        std::uint64_t get_num() { // based on George Marsaglia's xorshift
            x ^= x << 16;
            x ^= x >> 5;
            x ^= x << 1;

            std::uint64_t t = x;
            x = y;
            y = z;
            z = t ^ x ^ y;

            return z;
        }

        [[nodiscard]] int operator()(gya::board const &b) {
            auto idx = get_num() % 7;
            int iters = 0;
            while (b.data[idx].height == 6 && iters++ < 100) {
                idx = get_num() % 7;
            }
            if (b.data[idx].height == 6) {
                for (idx = 0; idx < 7; ++idx) {
                    if (b.data[idx].height < 6)
                        break;
                }
                return -1;
            }
            return idx;
        }
    };
}
