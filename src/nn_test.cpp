#pragma once

#include "include.hpp"

#include "heuristic/mcts/mcts.hpp"
#include "heuristic/brute_force/n_move_solver.hpp"
#include "heuristic/brute_force/one_move_solver.hpp"
#include "heuristic/brute_force/transposition_table_solver.hpp"
#include "heuristic/brute_force/two_move_solver.hpp"
#include "heuristic/solver_variations/A.hpp"
#include "heuristic/solver_variations/Abias.hpp"
#include "heuristic/solver_variations/simple_n_move_solver.hpp"

#include "../lib/tiny_dnn/tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

int main() {
    network<sequential> net;
    net << fully_connected_layer(42, 50)
        << tanh_layer()
        << fully_connected_layer(50, 50)
        << tanh_layer()
        << fully_connected_layer(50, 50)
        << tanh_layer()
        << fully_connected_layer(50, 7)
        << tanh_layer();


    const f32 learning_rate = 0.05f;
    const f32 discount_factor = 0.95f;

    heuristic::n_move_solver s(5);
    
    i32 iter = 5000;
    while(iter) {
        gya::board b;

        std::vector<vec_t> input_data;
        std::vector<vec_t> output_data;
        std::vector<u8> moves;
        std::vector<f32> qscore;

        u32 cnt = 0;
        i8 turn = 1;
        while(!b.has_won().is_game_over()) {
            if(!(iter % 2)) lmj::print(b.to_string());

            vec_t input;
            for(i32 i = gya::BOARD_HEIGHT - 1; i >= 0; i--) 
                for(u32 j = 0; j < gya::BOARD_WIDTH; j++) 
                    input.push_back(b[j][i]);

            input_data.push_back(input);

            vec_t result = net.predict(input);

            output_data.push_back(result);

            u32 move;
            f32 mx_output = -100;
            for(u32 col = 0; col < gya::BOARD_WIDTH; col++) 
                if(result[col] > mx_output && b[col].height < gya::BOARD_HEIGHT) {
                    mx_output = result[col];
                    move = col;
                }

            if(turn == -1) move = s(b);

            moves.push_back(move);

            if(cnt > 1) qscore.push_back(discount_factor * mx_output);

            b.play(move, turn);

            turn *= -1;
            cnt++;
        }

        if(b.has_won().is_tie()) {
            qscore.push_back(0);
            qscore.push_back(0);
        }
        else {
            qscore.push_back(-1);
            qscore.push_back(1);
        }

        if(!(iter % 2)) lmj::print(b.to_string());

        std::vector<vec_t> q_targets = output_data;

        for(u32 i = 0; i < cnt; i++) 
            q_targets[i][moves[i]] += learning_rate * (qscore[i] - q_targets[i][moves[i]]);

        size_t batch_size = 1;
        size_t epochs = 1;
        adagrad opt;

        net.fit<mse>(opt, input_data, q_targets, batch_size, epochs);

        if(!(iter % 2)) lmj::print(iter);

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
                lmj::timer t{false};

                vec_t input;
                for(i32 i = gya::BOARD_HEIGHT - 1; i >= 0; i--) 
                    for(u32 j = 0; j < gya::BOARD_WIDTH; j++) 
                        input.push_back(b[j][i]);

                vec_t result = net.predict(input);

                u32 move;
                f32 mx_output = -100;
                for(u32 col = 0; col < gya::BOARD_WIDTH; col++) 
                    if(result[col] > mx_output && b[col].height < gya::BOARD_HEIGHT) {
                        mx_output = result[col];
                        move = col;
                    }

                b.play(move);

                printf("%fs\n", t.elapsed());
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
        } else if (b.has_won().player_2_won()) {
            std::puts("you won!");
        } else {
            std::puts("you lost!");
        }
    }
}
