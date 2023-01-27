#include "include.hpp"

#include "heuristic/brute_force/n_move_solver.hpp"
#include "heuristic/brute_force/one_move_solver.hpp"
#include "heuristic/brute_force/transposition_table_solver.hpp"
#include "heuristic/brute_force/two_move_solver.hpp"
#include "heuristic/mcts/mcts.hpp"
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
    // int iter = 100000;
    // mcts::mcts s(10'000);
    // heuristic::n_move_solver heur{3, false};
    // {
    //     int wins = 0, ties = 0, losses = 0;
    //     for (int i = 0; i < 100; ++i) {
    //         lmj::print(i);
    //         gya::board b;
    //         const bool mcts_start = std::rand() % 2;
    //         bool turn = mcts_start;
    //         double mcts_time = 0, n_move_solver_time = 0;
    //         while (!b.has_won().is_game_over()) {
    //             if (turn == 1) {
    //                 lmj::timer t{false};
    //                 b.play(s.move(b, b.turn()));
    //                 mcts_time += t.elapsed();
    //             } else {
    //                 lmj::timer t{false};
    //                 b.play(heur(b));
    //                 n_move_solver_time += t.elapsed();
    //             }
    //             turn = !turn;
    //         }
    //         std::cerr << "mcts: " << mcts_time << "s\n";
    //         std::cerr << "n_move_solver_time: " << n_move_solver_time << "s\n";
    //         std::cout.flush();
    //         std::cerr.flush();
    //         wins += mcts_start ? b.has_won().player_1_won() : b.has_won().player_2_won();
    //         losses += mcts_start ? b.has_won().player_2_won() : b.has_won().player_1_won();
    //         ties += b.has_won().is_tie();
    //     }
    //     lmj::print(wins, ties, losses);
    // }
    // return 0;

    int depth;
    std::cout << "depth? ";
    std::cin >> depth;
    while (true) {
        gya::board b;
        i8 turn = -1;
        heuristic::transposition_table_solver s{depth};
        while (!b.has_won().is_game_over()) {
            // lmj::print((std::string) s.evaluate_board(b), s.m_ttable.size());
            if (turn == 1) {
                lmj::print(b.to_string());
                std::cout.flush();
                u8 move;
                {
                    lmj::timer t{false};
                    move = s(b);
                    printf("%fs\n", t.elapsed());
                }
                b.play(move);
            } else {
                lmj::print(b.to_string());
                std::cout.flush();
                int move;
                std::cin >> move;
                auto actions = b.get_actions();
                if (std::find(actions.begin(), actions.end(), move - 1) == actions.end()) {
                    std::cout << "invalid move, say another: ";
                    std::cin >> move;
                }
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
