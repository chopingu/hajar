#pragma once

#include "include.hpp"

#include "heuristic/brute_force/n_move_solver.hpp"
#include "heuristic/brute_force/one_move_solver.hpp"
#include "heuristic/brute_force/transposition_table_solver.hpp"
#include "heuristic/brute_force/two_move_solver.hpp"
#include "heuristic/mcts/mcts.hpp"
#include "heuristic/solver_variations/A.hpp"
#include "heuristic/solver_variations/Abias.hpp"
#include "heuristic/solver_variations/simple_n_move_solver.hpp"

#include "../lib/tiny_dnn/tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

// global variables, so sue me
int num_hidden_layers;
i32 program_iter = 1'000'000;
// i32 program_iter = 100;
i32 num_iters_ran = 0;
std::vector<int> hidden_layer_sizes;
std::string curr_date;

struct net_printer {
    network<sequential> const &net;

    ~net_printer() {
        std::ofstream out_file(curr_date + ".network_data");
        out_file << net;
        std::ofstream info_file(curr_date + ".info");
        info_file << "num hidden layers: " << num_hidden_layers << '\n';
        info_file << "hidden layer sizes: ";
        for (auto i: hidden_layer_sizes)
            info_file << i << ' ';
        info_file << '\n';
    }
};

int main() {

    { // put current time in curr_date variable
        // https://stackoverflow.com/questions/3673226/how-to-print-time-in-format-2009-08-10-181754-811
        char buffer[1024]{};
        time_t timer = time(NULL);
        strftime(buffer, 26, "%Y-%m-%d-%H-%M-%S", localtime(&timer));
        curr_date = buffer;
        lmj::debug(curr_date);
    }


    std::cout << "num hidden layers: ";
    std::cin >> num_hidden_layers;
    std::cout << "layer sizes:\n";
    hidden_layer_sizes.resize(num_hidden_layers);
    for (auto &i: hidden_layer_sizes) std::cin >> i;
    std::cout << "learning rate (was 0.01): ";
    f32 learning_rate;
    std::cin >> learning_rate;
    std::cout << "discount factor (was 0.9): ";
    f32 discount_factor;
    std::cin >> discount_factor;


    network<sequential> net;
    net_printer _net_printer{net};
    // net << fully_connected_layer(43, 200)
    //     << tanh_layer()
    //     << fully_connected_layer(200, 200)
    //     << tanh_layer()
    //     << fully_connected_layer(200, 200)
    //     << tanh_layer()
    //     << fully_connected_layer(200, 200)
    //     << tanh_layer()
    //     << fully_connected_layer(200, 7);
    net << fully_connected_layer(43, hidden_layer_sizes.front());
    for (int i = 1; i < num_hidden_layers; ++i) {
        net << tanh_layer();
        net << fully_connected_layer(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]);
    }
    net << tanh_layer();
    net << fully_connected_layer(hidden_layer_sizes.back(), 7);

    u32 win_cnt = 0;
    u32 draw_cnt = 0;
    u32 loss_cnt = 0;

    heuristic::n_move_solver s(5);

    std::ofstream win_loss_tie_data(curr_date + ".win_data");

    lmj::timer t;
    while (program_iter && t.elapsed() < 6 * 60 * 60) {
        lmj::debug(num_iters_ran);
        if (!(program_iter % 1)) {
            win_loss_tie_data << win_cnt << ' ';
            win_loss_tie_data << draw_cnt << ' ';
            win_loss_tie_data << loss_cnt << ' ';
            win_loss_tie_data << std::endl;
            win_cnt = 0;
            draw_cnt = 0;
            loss_cnt = 0;
        }

        gya::board b;

        std::vector<vec_t> input_data;
        std::vector<vec_t> output_data;
        std::vector<u8> moves;
        std::vector<f32> qscore;

        u32 cnt = 0;
        i8 turn = 1;
        while (!b.has_won().is_game_over()) {
            // if (!(program_iter % 2)) lmj::print(b.to_string());

            vec_t input;
            for (i32 i = gya::BOARD_HEIGHT - 1; i >= 0; i--)
                for (u32 j = 0; j < gya::BOARD_WIDTH; j++)
                    input.push_back(b[j][i]);

            input.push_back(turn);

            input_data.push_back(input);

            vec_t result = net.predict(input);

            output_data.push_back(result);

            u32 move = 0;
            f32 mx_output = -100;
            for (u32 col = 0; col < gya::BOARD_WIDTH; col++)
                if (result[col] > mx_output && b[col].height < gya::BOARD_HEIGHT) {
                    mx_output = result[col];
                    move = col;
                }

            if (turn == -1) {
                move = s(b);
                mx_output = result[move];
            }

            moves.push_back(move);

            if (cnt > 1) qscore.push_back(discount_factor * mx_output);

            b.play(move, turn);

            turn *= -1;
            cnt++;
        }

        if (b.has_won().is_tie()) {
            qscore.push_back(0);
            qscore.push_back(0);
            draw_cnt++;
        } else {
            qscore.push_back(-1);
            qscore.push_back(1);
            if (b.has_won().player_2_won())
                loss_cnt++;
            else
                win_cnt++;
        }

        std::vector<vec_t> q_targets = output_data;

        for (u32 i = 0; i < cnt; i++)
            q_targets[i][moves[i]] += learning_rate * (qscore[i] - q_targets[i][moves[i]]);

        size_t batch_size = 1;
        size_t epochs = 1;
        adagrad opt;

        net.fit<mse>(opt, input_data, q_targets, batch_size, epochs);
        program_iter--;
        num_iters_ran++;
    }

    win_loss_tie_data << win_cnt << ' ';
    win_loss_tie_data << draw_cnt << ' ';
    win_loss_tie_data << loss_cnt << ' ';
    win_loss_tie_data << std::endl;
}
