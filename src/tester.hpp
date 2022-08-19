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
        std::uint64_t state;

        random_player() {
            static std::uint64_t offs = 0;
            state = ++offs * (std::chrono::system_clock::now().time_since_epoch().count() & 0xffffffff);
        }

        std::uint64_t get_num() {
            return state = state * 6446109739 + 1743737587;
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
