#include "include.hpp"

#include "neural_net_testing/neural_net_player.hpp"
#include "neural_net_testing/neural_net_player_deep.hpp"

#include "heuristic/mcts.hpp"
#include "heuristic/n_move_solver.hpp"
#include "heuristic/one_move_solver.hpp"
#include "heuristic/solver_variations/A.hpp"
#include "heuristic/solver_variations/Abias.hpp"
#include "heuristic/transposition_table_solver.hpp"
#include "heuristic/two_move_solver.hpp"

#include "pinguml/layer/create_layer.hpp"
#include "pinguml/layer/fully_connected_layer.hpp"
#include "pinguml/layer/input_layer.hpp"
#include "pinguml/layer/layer_base.hpp"
#include "pinguml/utils/activation.hpp"
#include "pinguml/utils/cost.hpp"
#include "pinguml/utils/math.hpp"
#include "pinguml/utils/tensor.hpp"

int main() {
    while (true) {
        gya::board b;
        int turn = 1;
        std::cout << "solver depth? ";
        int depth;
        std::cin >> depth;
        heuristic::transposition_table_solver s{depth};
        while (!b.has_won().is_game_over()) {
            lmj::print((std::string) s.evaluate_board(b), s.m_ttable.size());
            if (turn ^= 1) {
                lmj::print(b.to_string());
                u8 move;
                {
                    lmj::timer t{false};
                    move = s(b);
                    printf("%fs\n", t.elapsed());
                }
                b.play(move);
            } else {
                lmj::print(b.to_string());
                int move;
                std::cin >> move;
                b.play(move - 1);
            }
        }
        lmj::print(b.to_string());
        lmj::print((std::string) s.evaluate_board(b));

        if (b.has_won().is_tie()) {
            std::puts("tie!");
        } else if (b.has_won().player_1_won()) {
            std::puts("you won!");
        } else {
            std::puts("you lost!");
        }
    }
}
