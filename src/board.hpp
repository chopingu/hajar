#pragma once

#include <array>
#include <string>
#include <string_view>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <chrono>
#include <type_traits>

#include "defines.hpp"

#include "../lib/lmj/src/containers/static_vector.hpp"

namespace gya {
struct game_result {
    i8 state{};

    constexpr bool player_1_won() const { return state == 1; }

    constexpr bool player_2_won() const { return state == 2; }

    constexpr bool is_game_over() const { return player_1_won() || player_2_won() || is_tie(); }

    constexpr bool is_tie() const { return state == -1; }

    constexpr bool operator==(gya::game_result other) const { return state == other.state; }
};

struct board_column {
    std::array<i8, 6> data{};
    i8 height{};

    constexpr i8 &push(i8 value) {
        if (height >= 6) {
            std::cout << std::flush;
            std::cerr << "invalid column to push into (column full)" << std::endl;
            exit(0);
        }
        data[height] = value;
        ++height;
        return data[height - 1];
    }

    constexpr i8 &operator[](u64 idx) {
        return data[idx];
    }

    constexpr i8 const &operator[](u64 idx) const {
        return data[idx];
    }

    constexpr bool operator==(board_column const &other) const {
        return height == other.height && data == other.data;
    }
};

struct board {
    std::array<board_column, 7> data{};
    gya::game_result winner{0};
    i8 size = 0;

    /*
requires function input to be formatted as such (same as provided by board::to_string()):
| | | | | | | |
| | | | | | | |
|X| | | | | | |
|O|X| | | | | |
|O|O|X| | | | |
|X|O|O|X| | | |
|1|2|3|4|5|6|7|
     */
    static constexpr auto from_string(std::string_view str) {
        assert(str.size() == 112);

        board b;

        for (int row = 6; row-- > 0;) {
            for (int col = 0; col < 7; ++col) {
                const auto curr = std::tolower(str[row * 16 + col * 2 + 1]);
                if (curr == 'x') {
                    b.play(col, 1);
                } else if (curr == 'o') {
                    b.play(col, -1);
                }
            }
        }

        return b;
    }

    constexpr board_column &operator[](u64 idx) {
        return data[idx];
    }

    constexpr board_column const &operator[](u64 idx) const {
        return data[idx];
    }

    constexpr i8 &play(u8 column, i8 value) {
        if (size == 42)
            throw std::runtime_error("cant play if board is full (possible tie)");
        if (value != 1 && value != -1)
            throw std::runtime_error("invalid m_player");
        if (data[column].height >= 6) {
            std::cout << std::flush;
            std::cerr << std::flush;
            std::cerr << to_string() << std::endl;
            std::cerr << "column: " << (int) column << std::endl;
        }
        auto &ret = data[column].push(value);
        ++size;

        int col = column;
        int row = data[column].height - 1;

        int hor = 1;
        if (col > 0 && data[col - 1][row] == value) {
            hor++;
            if (col > 1 && data[col - 2][row] == value) {
                hor++;
                if (col > 2 && data[col - 3][row] == value) {
                    hor++;
                }
            }
        }
        if (col < 6 && data[col + 1][row] == value) {
            hor++;
            if (col < 5 && data[col + 2][row] == value) {
                hor++;
                if (col < 4 && data[col + 3][row] == value) {
                    hor++;
                }
            }
        }
        if (hor >= 4) {
            winner.state = value == -1 ? 2 : 1;
            return ret;
        }
        int ver = 1;
        if (row > 0 && data[col][row - 1] == value) {
            ver++;
            if (row > 1 && data[col][row - 2] == value) {
                ver++;
                if (row > 2 && data[col][row - 3] == value) {
                    ver++;
                }
            }
        }
        if (ver >= 4) {
            winner.state = value == -1 ? 2 : 1;
            return ret;
        }

        int diag_tl = 1;
        if (col > 0 && row > 0 && data[col - 1][row - 1] == value) {
            diag_tl++;
            if (col > 1 && row > 1 && data[col - 2][row - 2] == value) {
                diag_tl++;
                if (col > 2 && row > 2 && data[col - 3][row - 3] == value) {
                    diag_tl++;
                }
            }
        }
        if (col < 6 && row < 5 && data[col + 1][row + 1] == value) {
            diag_tl++;
            if (col < 5 && row < 4 && data[col + 2][row + 2] == value) {
                diag_tl++;
                if (col < 4 && row < 3 && data[col + 3][row + 3] == value) {
                    diag_tl++;
                }
            }
        }
        if (diag_tl >= 4) {
            winner.state = value == -1 ? 2 : 1;
            return ret;
        }

        int diag_tr = 1;
        if (col > 0 && row < 5 && data[col - 1][row + 1] == value) {
            diag_tr++;
            if (col > 1 && row < 4 && data[col - 2][row + 2] == value) {
                diag_tr++;
                if (col > 2 && row < 3 && data[col - 3][row + 3] == value) {
                    diag_tr++;
                }
            }
        }
        if (col < 6 && row > 0 && data[col + 1][row - 1] == value) {
            diag_tr++;
            if (col < 5 && row > 1 && data[col + 2][row - 2] == value) {
                diag_tr++;
                if (col < 4 && row > 2 && data[col + 3][row - 3] == value) {
                    diag_tr++;
                }
            }
        }
        if (diag_tr >= 4) {
            winner.state = value == -1 ? 2 : 1;
            return ret;
        }

        if (size == 42) {
            winner.state = -1;
        }

        return ret;
    }

    constexpr i8 &play(u8 column) {
        return play(column, turn());
    }

    /**
     * @param column
     * @return board m_state after playing the given move
     */
    [[nodiscard]] constexpr gya::board play_copy(u8 column) const {
        gya::board result = *this;
        result.play(column);
        return result;
    }

    [[nodiscard]] lmj::static_vector<u8, 7> get_actions() const {
        if (size == 42)
            throw std::runtime_error("no actions if board is full (possible tie)");
        lmj::static_vector<u8, 7> res;
        for (u8 i = 0; i < 7; ++i)
            if (data[i].height < 6)
                res.push_back(i);
        return res;
    }

    [[nodiscard]] constexpr game_result has_won() const {
        return game_result{winner};
    }

    [[nodiscard]] constexpr game_result has_won_test() const {
        if (size < 7) return game_result{0};
        /*
        X
        X
        X
        X
         */
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j + 3 < 6; ++j) {
                if (data[i][j] == 0)
                    continue;
                if (data[i][j] == data[i][j + 1] &&
                    data[i][j] == data[i][j + 2] &&
                    data[i][j] == data[i][j + 3]) {
                    return data[i][j] == 1 ? game_result{1} : game_result{2};
                }
            }
        }


        /*
        X X X X
         */
        for (int i = 0; i + 3 < 7; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (data[i][j] == 0)
                    continue;
                if (data[i][j] == data[i + 1][j] &&
                    data[i][j] == data[i + 2][j] &&
                    data[i][j] == data[i + 3][j]) {
                    return data[i][j] == 1 ? game_result{1} : game_result{2};
                }
            }
        }

        /*
            X
           X
          X
         X
        */
        for (int i = 0; i + 3 < 7; ++i) {
            for (int j = 0; j + 3 < 6; ++j) {
                if (data[i][j] == 0)
                    continue;
                if (data[i][j] == data[i + 1][j + 1] &&
                    data[i][j] == data[i + 2][j + 2] &&
                    data[i][j] == data[i + 3][j + 3]) {
                    return data[i][j] == 1 ? game_result{1} : game_result{2};
                }
            }
        }

        /*
         X
          X
           X
            X
        */
        for (int i = 0; i + 3 < 7; ++i) {
            for (int j = 0; j + 3 < 6; ++j) {
                if (data[i][j + 3] == 0)
                    continue;
                if (data[i][j + 3] == data[i + 1][j + 2] &&
                    data[i][j + 3] == data[i + 2][j + 1] &&
                    data[i][j + 3] == data[i + 3][j]) {
                    return data[i][j + 3] == 1 ? game_result{1} : game_result{2};
                }
            }
        }

        if (std::all_of(data.begin(), data.end(), [](auto &col) { return col.height == 6; })) {
            return game_result{-1};
        } else {
            return game_result{0};
        }
    }

    [[nodiscard]] constexpr u32 n_in_a_row_counter(u8 n, i8 player) const { //m_player = 1 or -1
        return n_vertical_count(n, player) + n_horizontal_count(n, player) + n_top_right_diagonal_count(n, player) +
               (n_top_left_diagonal_count(n, player));
    }

    [[nodiscard]] constexpr u32 n_vertical_count(u8 n, i8 player) const {
        u32 n_in_a_rows = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j + n - 1 < 6; j++) {
                u8 counter = 0;
                if (data[i][j] != player)
                    continue;
                for (int k = 1; k <= n - 1; k++) {
                    if (data[i][j] == data[i][j + k])
                        counter++;
                    if (counter == n - 1)
                        n_in_a_rows++;
                }
            }
        }
        return n_in_a_rows;
    }

    [[nodiscard]] constexpr u32 n_horizontal_count(u8 n, i8 player) const {
        u32 n_in_a_rows = 0;
        for (int i = 0; i + n - 1 < 7; i++) {
            for (int j = 0; j < 6; j++) {
                u8 counter = 0;
                if (data[i][j] != player)
                    continue;
                for (int k = 1; k <= n - 1; k++) {
                    if (data[i][j] == data[i + k][j])
                        counter++;
                    if (counter == n - 1)
                        n_in_a_rows++;
                }
            }
        }
        return n_in_a_rows;
    }

    [[nodiscard]] constexpr u32 n_top_right_diagonal_count(u8 n, i8 player) const {
        u32 n_in_a_rows = 0;
        for (int i = 0; i + n - 1 < 7; i++) {
            for (int j = 0; j + n - 1 < 6; j++) {
                u8 counter = 0;
                if (data[i][j] != player)
                    continue;
                for (int k = 1; k <= n - 1; k++) {
                    if (data[i][j] == data[i + k][j + k])
                        counter++;
                    if (counter == n - 1)
                        n_in_a_rows++;
                }
            }
        }
        return n_in_a_rows;
    }

    [[nodiscard]] constexpr u32 n_top_left_diagonal_count(u8 n, i8 player) const {
        u32 n_in_a_rows = 0;
        for (int i = 0; i + n - 1 < 7; i++) {
            for (int j = 0; j + n - 1 < 6; j++) {
                u8 counter = 0;
                if (data[i][j + n - 1] != player)
                    continue;
                for (int k = 1; k <= n - 1; k++) {
                    if (data[i][j + n - 1] == data[i + k][j + n - 1 - k])
                        counter++;
                    if (counter == n - 1)
                        n_in_a_rows++;
                }
            }
        }
        return n_in_a_rows;
    }

    [[nodiscard]] std::string to_string() const {
        std::string ret;
        ret.reserve(112);
        for (int i = 6; i-- > 0;) {
            ret += '|';
            for (int j = 0; j < 7; ++j) {
                ret += data[j][i] == 0 ? ' ' : data[j][i] == 1 ? 'X' : 'O';
                ret += '|';
            }
            ret += '\n';
        }
        ret += '|';
        for (int j = 0; j < 7; ++j) {
            ret += j + 1 + '0';
            ret += '|';
        }
        ret += '\n';
        return ret;
    }

    constexpr bool operator==(board const &other) const {
        return size == other.size && winner == other.winner && data == other.data;
    }

    [[nodiscard]] constexpr i8 turn() const {
        return size % 2 ? -1 : 1;
    }
};

struct random_player {
    u64 x = 123456789, y = 362436069, z = 521288629;

    constexpr explicit random_player(u64 seed = -1) {
        if (std::is_constant_evaluated()) {
            set_seed(1);
        } else {
            construct(seed);
        }
    }

    void construct(u64 seed) {
        static u64 offs = 0;
        if (seed == -1ull)
            seed = ++offs * (std::chrono::system_clock::now().time_since_epoch().count() & 0xffffffff);
        set_seed(seed);
    }

    constexpr void set_seed(u64 seed) {
        u64 t = seed;
        x ^= t;
        y ^= (t >> 32) ^ (t << 32);
        z ^= (t >> 16) ^ (t << 48);

        for (int i = 0; i < 128; ++i)
            get_num();
    }

    constexpr u64 get_num() { // based on George Marsaglia's xorshift
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        u64 t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return z;
    }

    [[nodiscard]] constexpr u8 operator()(gya::board const &b) {
        u8 idx = get_num() % 7;
        int iters = 0;
        while (b.data[idx].height == 6 && iters++ < 100) {
            idx = get_num() % 7;
        }
        if (b.data[idx].height == 6) {
            for (idx = 0; idx < 7; ++idx) {
                if (b.data[idx].height < 6)
                    break;
            }
            return -1;
        }
        return idx;
    }
};

struct compressed_column {
private:
    u8 data{};
public:
    [[nodiscard]] constexpr u8 height() const {
        u8 res;
        for (res = 6; res >= 1; --res)
            if (data & (1 << (res - 1)))
                break;
        return res;
    }

    static constexpr gya::board_column decompress(gya::compressed_column c) {
        gya::board_column res{};
        if (!c.data)
            return res;

        res.height = c.height();

        const i8 highest_player = (c.data >> 7) ? 1 : -1;
        const i8 other_player = (c.data >> 7) ? -1 : 1;

        for (u8 i = 0; i < res.height; ++i) {
            res[i] = (c.data & (1 << i)) ? highest_player : other_player;
        }

        return res;
    }

    static constexpr gya::compressed_column compress(gya::board_column b) {
        gya::compressed_column res{};
        if (!b.height)
            return res;
        i8 highest_player = b[b.height - 1];
        res.data |= (highest_player == 1) << 7;
        for (u8 i = 0; i < b.height; ++i) {
            res.data |= (b[i] == highest_player) << i;
        }
        return res;
    }

    constexpr bool operator==(gya::compressed_column c) const {
        return data == c.data;
    }

    constexpr bool operator!=(gya::compressed_column c) const {
        return data != c.data;
    }
};

static_assert([] { // loop through all possible columns and verify they are compressed and decompressed properly
    for (int num_moves = 0; num_moves <= 6; ++num_moves) {
        for (int i = 0; i < (1 << num_moves); ++i) {
            gya::board_column c;
            for (int j = 0; j < num_moves; ++j) {
                c.push((i & (1 << j)) ? 1 : -1);
            }
            if (c != gya::compressed_column::decompress(gya::compressed_column::compress(c))) {
                return false;
            }
        }
    }
    return true;
}());

struct compressed_board {
private:
    std::array<compressed_column, 7> data;
public:
    constexpr compressed_board() = default;

    constexpr compressed_board(gya::compressed_board const &) = default;

    constexpr compressed_board(gya::board const &b) : compressed_board{compress(b)} {}

    constexpr operator gya::board() const {
        return decompress(*this);
    }

    static constexpr gya::board decompress(gya::compressed_board c) {
        gya::board res;
        for (u8 i = 0; i < 7; ++i) {
            res.data[i] = gya::compressed_column::decompress(c.data[i]);
            res.size += res.data[i].height;
        }
        res.winner = res.has_won_test();
        return res;
    }

    static constexpr gya::compressed_board compress(gya::board const &b) {
        gya::compressed_board res;
        for (u8 i = 0; i < 7; ++i) {
            res.data[i] = gya::compressed_column::compress(b.data[i]);
        }
        return res;
    }

    constexpr bool operator==(gya::compressed_board const &b) const {
        return data == b.data;
    }

    constexpr bool operator!=(gya::compressed_board const &b) const {
        return data != b.data;
    }
};

static_assert([] { // test some randomly generated games to make sure they are compressed and decompressed properly
    for (int i = 0; i < 100; ++i) {
        gya::random_player p1(i), p2(i + 10);
        gya::board b;
        b.play(p1(b));
        b.play(p2(b));

        b.play(p1(b));
        b.play(p2(b));

        b.play(p1(b));
        b.play(p2(b));

        b.play(p1(b));

        if (b != gya::compressed_board::decompress(gya::compressed_board::compress(b)))
            return false;
    }
    return true;
}());
} // namespace gya
