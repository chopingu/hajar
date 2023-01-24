#include "include.hpp"

#include "heuristic/mcts/mcts.hpp"
#include "heuristic/brute_force/n_move_solver.hpp"
#include "heuristic/brute_force/one_move_solver.hpp"
#include "heuristic/brute_force/transposition_table_solver.hpp"
#include "heuristic/brute_force/two_move_solver.hpp"
#include "heuristic/solver_variations/A.hpp"
#include "heuristic/solver_variations/Abias.hpp"
#include "heuristic/solver_variations/simple_n_move_solver.hpp"

/*
#include "neural_net_testing/neural_net_player.hpp"
#include "neural_net_testing/neural_net_player_deep.hpp"

#include "pinguml/layer/create_layer.hpp"
#include "pinguml/layer/fully_connected_layer.hpp"
#include "pinguml/layer/input_layer.hpp"
#include "pinguml/layer/layer_base.hpp"
#include "pinguml/network.hpp"
#include "pinguml/utils/activation.hpp"
#include "pinguml/utils/cost.hpp"
#include "pinguml/utils/math.hpp"
#include "pinguml/utils/solver.hpp"
#include "pinguml/utils/tensor.hpp"
*/

int main() {
    int iter = 100000;
    mcts::mcts s(100);
    heuristic::transposition_table_solver heur(6);
    while(iter) {
        gya::board b;
        i8 turn = 1 - 2 * (std::rand() % 2);

        while(!b.has_won().is_game_over()) {
            if(!(iter % 10)) {
                lmj::print(b.to_string());
            }

            if(turn == 1) {
                b.play(s.move(b, turn));
            }
            else {
                b.play(heur(b));
            }
            turn *= -1;
        }
        if(!(iter % 10)) lmj::print(iter);

        iter--;
    }

    while (true) {
        gya::board b;
        i8 turn = 1;
        // heuristic::transposition_table_solver s{depth};
        while (!b.has_won().is_game_over()) {
            // lmj::print((std::string) s.evaluate_board(b), s.m_ttable.size());
            if (turn == 1) {
                lmj::print(b.to_string());
                u8 move;
                {
                    lmj::timer t{false};
                    move = s.move(b, turn);
                    printf("%fs\n", t.elapsed());
                }
                b.play(move);
            } else {
                lmj::print(b.to_string());
                int move;
                std::cin >> move;
                b.play(move - 1);
            }
            turn *= -1;
        }
        lmj::print(b.to_string());
        // lmj::print((std::string) s.evaluate_board(b));

        if (b.has_won().is_tie()) {
            std::puts("tie!");
        } else if (b.has_won().player_1_won()) {
            std::puts("you won!");
        } else {
            std::puts("you lost!");
        }
    }
}
