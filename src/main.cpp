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
#include "pinguml/network.hpp"
#include "pinguml/utils/activation.hpp"
#include "pinguml/utils/cost.hpp"
#include "pinguml/utils/math.hpp"
#include "pinguml/utils/solver.hpp"
#include "pinguml/utils/tensor.hpp"

int main() {
    while (true) {
        gya::board b;
        int turn = 1;
        std::cout << "solver depth? ";
        u32 depth;
        std::cin >> depth;
        // heuristic::transposition_table_solver s{depth};
        heuristic::A s{depth, 3};
        while (!b.has_won().is_game_over()) {
            // lmj::print((std::string) s.evaluate_board(b), s.m_ttable.size());
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
        // lmj::print((std::string) s.evaluate_board(b));

        if (b.has_won().is_tie()) {
            std::puts("tie!");
        } else if (b.has_won().player_1_won()) {
            std::puts("you won!");
        } else {
            std::puts("you lost!");
        }
    }

    pinguml::network nn("sgd");
//    nn.enable_threads(20);
    nn.set_batch_size(24);

    nn.push_back("I1", "input 1 1 1");
    nn.push_back("H1", "fully_connected 10 tanh");
    nn.push_back("H2", "fully_connected 10 tanh");
    //    nn.push_back("H2", "fully_connected 15 tanh");
    //    nn.push_back("H3", "fully_connected 20 tanh");
    //    nn.push_back("H4", "fully_connected 15 tanh");
    nn.push_back("H5", "fully_connected 1 tanh");

    nn.connect();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<f32> dist(-4, 4);

    std::vector<f32> x, y;
    for (u32 i = 0; i < 1000000; i++) {
        f32 val = dist(gen);
        x.push_back(val);
        y.push_back(std::sin(val));
    }

    int iter = 0;
    while (iter++ < 100) {
        lmj::debug(iter);
        std::cerr.flush();
        nn.start_epoch("mse");

//#pragma omp parallel
//#pragma omp for schedule(dynamic)
        for (u32 i = 0; i < (u32) x.size(); i++) {
            nn.train_target(&x[i], &y[i]);
        }

        nn.end_epoch();

        if (nn.over()) break;
    }

    //    for (u32 i = 0; i < x.size(); i++)
    //        std::cout << x[i] << ' ' << nn.forward(&x[i])[0] << "\n";
    {
        for (f32 i = -3; i <= 3; i += 0.1)
            //            std::cout << i << ' ' << nn.forward(&i)[0] << ' ' << std::sin(i) << ' ' << std::sin(i) - nn.forward(&i)[0] << '\n';
            std::cout << "\\left(" << i << "," << nn.forward(&i)[0] << "\\right)" << '\n';
        ;
    }
}
