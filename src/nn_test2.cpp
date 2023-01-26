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
    net << fully_connected_layer(43, 200)
        << tanh_layer()
        << fully_connected_layer(200, 200)
        << tanh_layer()
        << fully_connected_layer(200, 200)
        << tanh_layer()
        << fully_connected_layer(200, 200)
        << tanh_layer()
        << fully_connected_layer(200, 7);

    const f32 learning_rate = 0.1f;
    const f32 discount_factor = 0.9f;

    std::vector<u32> wins;
    std::vector<u32> draws;
    std::vector<u32> losses;

    u32 win_cnt;
    u32 draw_cnt;
    u32 loss_cnt;

    heuristic::n_move_solver s(5);

    i32 iter = 10000;
    while(iter) {
        if(!(iter % 100)) {
            wins.push_back(win_cnt);
            draws.push_back(draw_cnt);
            losses.push_back(loss_cnt);
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
        while(!b.has_won().is_game_over()) {
            if(!(iter % 2)) lmj::print(b.to_string());

            vec_t input;
            for(i32 i = gya::BOARD_HEIGHT - 1; i >= 0; i--) 
                for(u32 j = 0; j < gya::BOARD_WIDTH; j++) 
                    input.push_back(b[j][i]);

            input.push_back(turn);

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

            if(turn == -1) {
                move = s(b);
                mx_output = result[move];
            }

            moves.push_back(move);

            if(cnt > 1) qscore.push_back(discount_factor * mx_output);

            b.play(move, turn);

            turn *= -1;
            cnt++;
        }

        if(b.has_won().is_tie()) {
            qscore.push_back(0);
            qscore.push_back(0);
            draw_cnt++;
        }
        else {
            qscore.push_back(-1);
            qscore.push_back(1);
            if(b.has_won().player_2_won()) 
                loss_cnt++;
            else 
                win_cnt++;
        }

        std::vector<vec_t> q_targets = output_data;

        for(u32 i = 0; i < cnt; i++) 
            q_targets[i][moves[i]] += learning_rate * (qscore[i] - q_targets[i][moves[i]]);

        size_t batch_size = 30;
        size_t epochs = 100;
        adagrad opt;

        net.fit<mse>(opt, input_data, q_targets, batch_size, epochs);

        iter--;
    }

    wins.push_back(win_cnt);
    draws.push_back(draw_cnt);
    losses.push_back(loss_cnt);
}
