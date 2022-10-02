#include "../defines.hpp"
#include "../board.hpp"
#include "../tester.hpp"

namespace gya {

struct one_move_solver {
    gya::random_player rp{};

    u8 operator()(gya::board const &b) {
        i8 turn = b.size % 2 ? 1 : -1;
        for (u8 i = 0; i < 7; ++i) {
            gya::board copy = b;
            if (copy.data[i].height == 6)
                continue;
            copy.play(i, turn);
            if (copy.has_won().state == turn)
                return i;
        }
        return rp(b);
    }
};
}
