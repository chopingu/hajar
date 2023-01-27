#include "heuristic/n_move_solver.hpp"
#include "heuristic/transposition_table_solver.hpp"
#include "include.hpp"
#include "mcts/mcts.hpp"
#include "tiny_dnn/tiny_dnn.h"

int main() {
    heuristic::n_move_solver corr;

    tiny_dnn::network<tiny_dnn::sequential> net;
    
    net << tiny_dnn::fully_connected_layer(42, 32);
    net << tiny_dnn::tanh_layer();
    net << tiny_dnn::fully_connected_layer(32, 16);
    net << tiny_dnn::tanh_layer();
    net << tiny_dnn::fully_connected_layer(16, 7);

    tiny_dnn::adamax opt;

    constexpr int num_training_games = 200'000;

    tiny_dnn::tensor_t input(num_training_games, tiny_dnn::), output(num_training_games);

    { // generate training data
        gya::random_player rp;
        heuristic::transposition_table_solver brute{.m_depth = 8};

    }
}