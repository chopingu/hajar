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

int kernel_size;
f32 learning_rate;
int random_rate;
const std::string mode = "DQ";
const std::string path = "../data/";
std::string curr_date;
std::string identifier;

struct net_printer {
    network<sequential> const &net;

    ~net_printer() {
        std::ofstream out_file(path + identifier + ".network_data");
        out_file << net;
    }
};

void benchmark(network<sequential> &net) {
    struct solver_from_network {
        network<sequential> &net;

        u8 operator()(gya::board &b) {
            vec_t input;
            for (i32 i = gya::BOARD_HEIGHT - 1; i >= 0; i--)
                for (i32 j = 0; j < gya::BOARD_WIDTH; j++)
                    input.push_back(b[j][i]);
            vec_t result = net.predict(input);
            f32 mx_output = -100;
            u8 move = 0;
            for (u32 col = 0; col < gya::BOARD_WIDTH; col++) {
                if (result[col] > mx_output && b[col].height < gya::BOARD_HEIGHT) {
                    mx_output = result[col];
                    move = col;
                }
            }
            return move;
        }
    };
    solver_from_network solver{net};
    using heur_func = u8(const gya::board &);
    static const std::pair<heur_func *, const char *> heuristics[]{
            {+[](gya::board const &board) { return heuristic::n_move_solver{4}(board); }, "n_move_solver_4"},
            {+[](gya::board const &board) { auto solver = mcts::mcts{500}; return solver.move(board, board.turn()); }, "mcts500"},
            {+[](gya::board const &board) { auto solver = mcts::mcts{1000}; return solver.move(board, board.turn()); }, "mcts1000"},
            {+[](gya::board const &board) { auto solver = mcts::mcts{2000}; return solver.move(board, board.turn()); }, "mcts2000"},
            {+[](gya::board const &board) { return heuristic::two_move_solver{}(board); }, "two_move_solver"},
    };

    static std::vector<std::ofstream> outfiles = [&] {
        std::vector<std::ofstream> result;
        for (auto [_, name]: heuristics)
            result.emplace_back(path + identifier + "-" + name + ".benchmark_data");
        return result;
    }();


    for (usize i = 0; i < std::size(heuristics); ++i) {
        i32 w = 0, t = 0, l = 0;
        for (int game_idx = 0; game_idx < 50; ++game_idx) {
            gya::board b1 = util::test_game(solver, heuristics[i].first);
            gya::board b2 = util::test_game(heuristics[i].first, solver);

            w += b1.has_won().player_1_won();
            t += b1.has_won().is_tie();
            l += b1.has_won().player_2_won();

            l += b2.has_won().player_1_won();
            t += b2.has_won().is_tie();
            w += b2.has_won().player_2_won();
        }
        outfiles[i] << w << ' ' << t << ' ' << l << std::endl;
    }
}

int main() {
    { // put current time in curr_date variable
        // https://stackoverflow.com/questions/3673226/how-to-print-time-in-format-2009-08-10-181754-811
        char buffer[1024]{};
        time_t timer = time(NULL);
        strftime(buffer, 26, "%Y-%m-%d-%H-%M-%S", localtime(&timer));
        curr_date = buffer;
        lmj::debug(curr_date);
    }

    std::cout << "kernel_size: ";
    std::cin >> kernel_size;
    std::cout << "learning_rate: ";
    std::cin >> learning_rate;
    std::cout << "random_rate: ";
    std::cin >> random_rate;
    identifier = mode + "-K" + std::to_string(kernel_size) + "-L" + std::to_string(static_cast<i32>(learning_rate * 1000)) + "-R" + std::to_string(random_rate) + "-" + curr_date;
    lmj::debug(identifier);

    size_t in_width = 7;
    size_t in_height = 6;
    size_t window_size = kernel_size;
    size_t in_channels = 1;
    size_t out_channels = 256;
    network<sequential> net;
    net_printer _net_printer(net);
    net << convolutional_layer(in_width, in_height, window_size, in_channels, out_channels)
        << tanh_layer(in_width - window_size + 1, in_height - window_size + 1, out_channels)
        << fully_connected_layer((in_width - window_size + 1) * (in_height - window_size + 1) * out_channels, 64)
        << tanh_layer()
        << fully_connected_layer(64, 64)
        << tanh_layer()
        << fully_connected_layer(64, 7);

    network<sequential> net2;
    net2 << convolutional_layer(in_width, in_height, window_size, in_channels, out_channels)
         << tanh_layer(in_width - window_size + 1, in_height - window_size + 1, out_channels)
         << fully_connected_layer((in_width - window_size + 1) * (in_height - window_size + 1) * out_channels, 64)
         << tanh_layer()
         << fully_connected_layer(64, 64)
         << tanh_layer()
         << fully_connected_layer(64, 7);

    const f32 discount_factor = 0.9f;

    u32 win_cnt = 0;
    u32 draw_cnt = 0;
    u32 loss_cnt = 0;

    std::ofstream win_loss_tie_data(path + identifier + ".win_data");

    i32 iter = 10000;
    lmj::timer timer{false};
    i32 hours = 5;
    while (timer.elapsed() < hours * 60 * 60) {
        if (iter % 100 == 0)
            benchmark(net);
        lmj::debug(timer.elapsed());
        lmj::print(iter);
        if (!(iter % 1)) {
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
        std::vector<u8> moves;
        std::vector<i8> col_times(7, -1);

        u32 cnt = 0;
        i8 turn = 1;
        while (!b.has_won().is_game_over()) {
            if (!(iter % 2)) lmj::print(b.to_string());

            vec_t input;
            for (i32 i = gya::BOARD_HEIGHT - 1; i >= 0; i--)
                for (u32 j = 0; j < gya::BOARD_WIDTH; j++)
                    input.push_back(b[j][i]);

            input_data.push_back(input);

            vec_t result = net.predict(input);
            vec_t result2 = net2.predict(input);

            u32 move = 0, move2 = 0;
            std::srand(std::clock());
            if (std::rand() % random_rate) {
                f32 mx_output = -100;
                f32 mx_output2 = -100;
                for (u32 col = 0; col < gya::BOARD_WIDTH; col++) {
                    if (result[col] > mx_output && b[col].height < gya::BOARD_HEIGHT) {
                        mx_output = result[col];
                        move = col;
                    }
                    if (result2[col] > mx_output2 && b[col].height < gya::BOARD_HEIGHT) {
                        mx_output2 = result2[col];
                        move2 = col;
                    }
                    if (b[col].height >= gya::BOARD_HEIGHT && col_times[col] == -1)
                        col_times[col] = cnt;
                }

                if(std::rand() % 2) 
                    move = move2;
            } else {
                auto legal = b.get_actions();
                move = legal[std::rand() % legal.size()];
            }

            moves.push_back(move);

            b.play(move, turn);

            turn *= -1;
            cnt++;
        }

        for (auto &i: col_times)
            if (i != -1)
                i = cnt - i - 1;

        f32 qlast = 0, qnlast = 0;
        f32 q2last = 0, q2nlast = 0;

        if (b.has_won().is_tie()) {
            draw_cnt++;
        } else {
            qlast = 1, qnlast = -1;
            q2last = 1, q2nlast = -1;
            if (b.has_won().player_2_won())
                loss_cnt++;
            else
                win_cnt++;
        }

        adagrad opt;

        std::reverse(moves.begin(), moves.end());
        std::reverse(input_data.begin(), input_data.end());

        for (u32 i = 0; i < cnt; i++) {
            vec_t q = net.predict(input_data[i]);
            vec_t q2 = net2.predict(input_data[i]);
            q[moves[i]] += learning_rate * (qlast - q[moves[i]]);
            q2[moves[i]] += learning_rate * (q2last - q2[moves[i]]);

            net.fit<mse>(opt, std::vector<vec_t>{input_data[i]}, std::vector<vec_t>{q});
            net2.fit<mse>(opt, std::vector<vec_t>{input_data[i]}, std::vector<vec_t>{q2});

            q = net.predict(input_data[i]);
            q2 = net2.predict(input_data[i]);

            qlast = qnlast;
            q2last = q2nlast;

            f32 mx_q = -100;
            f32 mx_q2 = -100;
            u32 move = 0, move2 = 0;
            for (u32 j = 0; j < gya::BOARD_WIDTH; j++)
                if (col_times[j] < (i8) i) {
                    if(mx_q < q[j]) {
                        mx_q = q[j];
                        move = j;
                    }
                    if(mx_q2 < q2[j]) {
                        mx_q2 = q2[j];
                        move2 = j;
                    }
                }

            qnlast = discount_factor * q2[move];
            q2nlast = discount_factor * q[move2];
        }

        lmj::debug(input_data.size());

        iter--;
    }

    win_loss_tie_data << win_cnt << ' ';
    win_loss_tie_data << draw_cnt << ' ';
    win_loss_tie_data << loss_cnt << ' ';
    win_loss_tie_data << std::endl;
}
