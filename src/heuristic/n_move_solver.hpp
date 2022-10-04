#pragma once

#include "../defines.hpp"
#include "../board.hpp"
#include "../tester.hpp"

namespace heuristic {
    struct n_move_solver {
        gya::random_player rp{};
        u8 operator()(gya::board const &b) {
            return rp(b);
        }
    };
}
