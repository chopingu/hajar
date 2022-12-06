#pragma once

#include "../lib/lmj/src/include_all.hpp"
#include "defines.hpp"

namespace gya {

constexpr auto BOARD_HEIGHT = 6;
constexpr auto BOARD_WIDTH = 7;

struct game_result {
    static constexpr i8 GAME_NOT_OVER = 0;
    static constexpr i8 PLAYER_ONE_WON = 1;
    static constexpr i8 PLAYER_TWO_WON = 2;
    static constexpr i8 TIE = -1;

    constexpr game_result(i8 v) : state{v} {}

    i8 state = GAME_NOT_OVER;

    [[nodiscard]] constexpr bool player_1_won() const { return state == PLAYER_ONE_WON; }

    [[nodiscard]] constexpr bool player_2_won() const { return state == PLAYER_TWO_WON; }

    [[nodiscard]] constexpr bool is_game_over() const { return state != GAME_NOT_OVER; }

    [[nodiscard]] constexpr bool is_tie() const { return state == TIE; }

    [[nodiscard]] constexpr bool operator==(gya::game_result const &other) const = default;
};

struct board_column {
    std::array<i8, BOARD_WIDTH> data{};
    u8 height{};

    constexpr i8 &push(i8 value) {
        if (height >= BOARD_HEIGHT) {
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

    constexpr bool operator==(board_column const &other) const = default;
};

struct board {
    static constexpr i8 PLAYER_ONE = 1;
    static constexpr i8 PLAYER_TWO = -1;

    std::array<board_column, BOARD_WIDTH> data{};
    gya::game_result winner{gya::game_result::GAME_NOT_OVER};
    u8 size = 0;

    constexpr i8 &play(u8 column, i8 value) {
        if (size == BOARD_WIDTH * BOARD_HEIGHT)
            throw std::runtime_error("cant play if board is full (possible tie)");
        if (value != PLAYER_ONE && value != PLAYER_TWO)
            throw std::runtime_error("invalid m_player");
        if (data[column].height >= BOARD_HEIGHT) {
            std::cout.flush();
            std::cerr.flush();
            std::cerr << to_string() << '\n';
            std::cerr << "column: " << (int) (column + 1) << " (one-indexed)\n";
            std::cerr.flush();
        }

        if (const auto res = is_winning_move(column, value); res != winner) {
            winner = res;
        }

        auto &ret = data[column].push(value);
        ++size;

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

    [[nodiscard]] constexpr gya::game_result is_winning_move(u8 column, i8 value) const {
        int const col = column;
        int const row = data[column].height;

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
        if (col + 1 < BOARD_WIDTH && data[col + 1][row] == value) {
            hor++;
            if (col + 2 < BOARD_WIDTH && data[col + 2][row] == value) {
                hor++;
                if (col + 3 < BOARD_WIDTH && data[col + 3][row] == value) {
                    hor++;
                }
            }
        }
        if (hor >= 4) {
            return value == PLAYER_ONE ? game_result::PLAYER_ONE_WON : game_result::PLAYER_TWO_WON;
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
            return value == PLAYER_ONE ? game_result::PLAYER_ONE_WON : game_result::PLAYER_TWO_WON;
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
        if (col + 1 < BOARD_WIDTH && row + 1 < BOARD_HEIGHT && data[col + 1][row + 1] == value) {
            diag_tl++;
            if (col + 2 < BOARD_WIDTH && row + 2 < BOARD_HEIGHT && data[col + 2][row + 2] == value) {
                diag_tl++;
                if (col + 3 < BOARD_WIDTH && row + 3 < BOARD_HEIGHT && data[col + 3][row + 3] == value) {
                    diag_tl++;
                }
            }
        }
        if (diag_tl >= 4) {
            return value == PLAYER_ONE ? game_result::PLAYER_ONE_WON : game_result::PLAYER_TWO_WON;
        }

        int diag_tr = 1;
        if (col > 0 && row + 1 < BOARD_HEIGHT && data[col - 1][row + 1] == value) {
            diag_tr++;
            if (col > 1 && row + 2 < BOARD_HEIGHT && data[col - 2][row + 2] == value) {
                diag_tr++;
                if (col > 2 && row + 3 < BOARD_HEIGHT && data[col - 3][row + 3] == value) {
                    diag_tr++;
                }
            }
        }
        if (col + 1 < BOARD_WIDTH && row > 0 && data[col + 1][row - 1] == value) {
            diag_tr++;
            if (col + 2 < BOARD_WIDTH && row > 1 && data[col + 2][row - 2] == value) {
                diag_tr++;
                if (col + 3 < BOARD_WIDTH && row > 2 && data[col + 3][row - 3] == value) {
                    diag_tr++;
                }
            }
        }
        if (diag_tr >= 4) {
            return value == PLAYER_ONE ? game_result::PLAYER_ONE_WON : game_result::PLAYER_TWO_WON;
        }

        if (size + 1 == BOARD_WIDTH * BOARD_HEIGHT) {
            return game_result::TIE;
        } else {
            return game_result::GAME_NOT_OVER;
        }
    }

    [[nodiscard]] constexpr gya::game_result is_winning_move(u8 column) const {
        return is_winning_move(column, turn());
    }

    [[nodiscard]] lmj::static_vector<u8, BOARD_WIDTH> get_actions() const {
        lmj::static_vector<u8, BOARD_WIDTH> res;
        for (u8 i = 0; i < BOARD_WIDTH; ++i)
            if (data[i].height < BOARD_HEIGHT)
                res.push_back(i);
        return res;
    }

    [[nodiscard]] constexpr game_result has_won() const {
        return winner;
    }

    [[nodiscard]] constexpr game_result has_won_test() const {
        if (size < BOARD_WIDTH) return game_result::GAME_NOT_OVER;
        /*
        X
        X
        X
        X
         */
        for (int i = 0; i < BOARD_WIDTH; ++i) {
            for (int j = 0; j + 3 < BOARD_HEIGHT; ++j) {
                if (data[i][j] == 0)
                    continue;
                if (data[i][j] == data[i][j + 1] &&
                    data[i][j] == data[i][j + 2] &&
                    data[i][j] == data[i][j + 3]) {
                    return data[i][j] == PLAYER_ONE ? game_result::PLAYER_ONE_WON : game_result::PLAYER_TWO_WON;
                }
            }
        }


        /*
        X X X X
         */
        for (int i = 0; i + 3 < BOARD_WIDTH; ++i) {
            for (int j = 0; j < BOARD_HEIGHT; ++j) {
                if (data[i][j] == 0)
                    continue;
                if (data[i][j] == data[i + 1][j] &&
                    data[i][j] == data[i + 2][j] &&
                    data[i][j] == data[i + 3][j]) {
                    return data[i][j] == PLAYER_ONE ? game_result::PLAYER_ONE_WON : game_result::PLAYER_TWO_WON;
                }
            }
        }

        /*
            X
           X
          X
         X
        */
        for (int i = 0; i + 3 < BOARD_WIDTH; ++i) {
            for (int j = 0; j + 3 < BOARD_HEIGHT; ++j) {
                if (data[i][j] == 0)
                    continue;
                if (data[i][j] == data[i + 1][j + 1] &&
                    data[i][j] == data[i + 2][j + 2] &&
                    data[i][j] == data[i + 3][j + 3]) {
                    return data[i][j] == PLAYER_ONE ? game_result::PLAYER_ONE_WON : game_result::PLAYER_TWO_WON;
                }
            }
        }

        /*
         X
          X
           X
            X
        */
        for (int i = 0; i + 3 < BOARD_WIDTH; ++i) {
            for (int j = 0; j + 3 < BOARD_HEIGHT; ++j) {
                if (data[i][j + 3] == 0)
                    continue;
                if (data[i][j + 3] == data[i + 1][j + 2] &&
                    data[i][j + 3] == data[i + 2][j + 1] &&
                    data[i][j + 3] == data[i + 3][j]) {
                    return data[i][j + 3] == PLAYER_ONE ? game_result::PLAYER_ONE_WON : game_result::PLAYER_TWO_WON;
                }
            }
        }

        if (std::all_of(data.begin(), data.end(), [](auto &col) { return col.height == 6; })) {
            return game_result::TIE;
        } else {
            return game_result::GAME_NOT_OVER;
        }
    }

    [[nodiscard]] constexpr u32 n_in_a_row_counter(u8 n, i8 player) const { // player = 1 or -1
        return n_vertical_count(n, player) + n_horizontal_count(n, player) + n_top_right_diagonal_count(n, player) +
               (n_top_left_diagonal_count(n, player));
    }

    [[nodiscard]] constexpr u32 n_vertical_count(u8 n, i8 player) const {
        u32 n_in_a_rows = 0;
        for (int i = 0; i < BOARD_WIDTH; i++) {
            for (int j = 0; j + n - 1 < BOARD_HEIGHT; j++) {
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
        for (int i = 0; i + n - 1 < BOARD_WIDTH; i++) {
            for (int j = 0; j < BOARD_HEIGHT; j++) {
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
        for (int i = 0; i + n - 1 < BOARD_WIDTH; i++) {
            for (int j = 0; j + n - 1 < BOARD_HEIGHT; j++) {
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
        for (int i = 0; i + n - 1 < BOARD_WIDTH; i++) {
            for (int j = 0; j + n - 1 < BOARD_HEIGHT; j++) {
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

    /**
     * requires function input to be formatted as such (same as provided by board::to_string()):
     * | | | | | | | |
     * | | | | | | | |
     * |X| | | | | | |
     * |O|X| | | | | |
     * |O|O|X| | | | |
     * |X|O|O|X| | | |
     * |1|2|3|4|5|6|7|
     */
    static constexpr auto from_string(std::string_view str) {
        [[maybe_unused]] constexpr auto NUM_CHARS_REQUIRED = 112;
        assert(str.size() == NUM_CHARS_REQUIRED);

        board b;

        for (int row = BOARD_HEIGHT; row-- > 0;) {
            for (int col = 0; col < BOARD_WIDTH; ++col) {
                const auto curr = std::tolower(str[row * 16 + col * 2 + 1]);
                if (curr == 'x') {
                    b.play(col, PLAYER_ONE);
                } else if (curr == 'o') {
                    b.play(col, PLAYER_TWO);
                }
            }
        }

        return b;
    }

    [[nodiscard]] std::string to_string() const {
        std::string ret;
        constexpr auto NUM_CHARS_REQUIRED = 112;
        ret.reserve(NUM_CHARS_REQUIRED);
        for (usize i = BOARD_HEIGHT; i-- > 0;) {
            ret += '|';
            for (usize j = 0; j < BOARD_WIDTH; ++j) {
                ret += data[j][i] == 0 ? ' ' : data[j][i] == PLAYER_ONE ? 'X' : 'O';
                ret += '|';
            }
            ret += '\n';
        }
        ret += '|';
        for (usize j = 0; j < BOARD_WIDTH; ++j) {
            ret += char(j + 1 + '0');
            ret += '|';
        }
        ret += '\n';
        return ret;
    }

    constexpr board_column &operator[](u64 idx) {
        return data[idx];
    }

    constexpr board_column const &operator[](u64 idx) const {
        return data[idx];
    }

    constexpr bool operator==(board const &other) const = default;

    [[nodiscard]] constexpr i8 turn() const {
        return size % 2 ? PLAYER_TWO : PLAYER_ONE;
    }
};

struct random_player {
private:
    constexpr static auto SEEDS = std::array{123456789ull, 362436069ull, 521288629ull};

    u64 x = SEEDS[0], y = SEEDS[1], z = SEEDS[2];
public:
    constexpr explicit random_player(u64 seed = -1) {
        if (std::is_constant_evaluated())
            set_seed(1);
        else
            construct(seed);
    }

    void construct(u64 seed) {
        static u64 offs = 0;
        if (seed == -1ull)
            seed = ++offs * (std::chrono::system_clock::now().time_since_epoch().count() & 0xffffffff);
        set_seed(seed);
    }

    constexpr void set_seed(u64 seed) {
        x ^= seed;
        y ^= (seed >> 32) ^ (seed << 32);
        z ^= (seed >> 16) ^ (seed << 48);

        constexpr auto NUM_DISCARDED_VALUES = 128;
        for (int i = 0; i < NUM_DISCARDED_VALUES; ++i)
            get_num();
    }

    constexpr u64 get_num() { // based on George Marsaglia's xorshift
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        u64 const t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return z;
    }

    [[nodiscard]] constexpr u8 operator()(gya::board const &b) {
        u8 idx = get_num() % BOARD_WIDTH;
        usize iters = 0;
        constexpr auto NUM_RANDOM_TRIES = 1 << 7;
        while (b.data[idx].height == BOARD_HEIGHT && iters++ < NUM_RANDOM_TRIES)
            idx = get_num() % BOARD_WIDTH;
        if (b.data[idx].height == BOARD_HEIGHT) {
            for (idx = 0; idx < BOARD_WIDTH; ++idx)
                if (b.data[idx].height < BOARD_HEIGHT)
                    break;
            return -1;
        }
        return idx;
    }
};

struct compressed_column {
private:
    u8 data{};
public:
    constexpr static auto TURN_BIT_POS = 7;

    [[nodiscard]] constexpr u8 height() const {
        u8 res = 0;
        for (res = BOARD_HEIGHT; res >= 1; --res)
            if (data & (1 << (res - 1)))
                break;
        return res;
    }

    static constexpr gya::board_column decompress(gya::compressed_column c) {
        gya::board_column res{};
        if (!c.data)
            return res;

        res.height = c.height();

        const i8 highest_player = (c.data >> TURN_BIT_POS) ? board::PLAYER_ONE : board::PLAYER_TWO;
        const i8 other_player = (c.data >> TURN_BIT_POS) ? board::PLAYER_TWO : board::PLAYER_ONE;

        for (u8 i = 0; i < res.height; ++i)
            res[i] = (c.data & (1 << i)) ? highest_player : other_player;

        return res;
    }

    static constexpr gya::compressed_column compress(gya::board_column b) {
        gya::compressed_column res{};
        if (!b.height)
            return res;
        i8 const highest_player = b[b.height - 1];
        res.data |= (highest_player == 1) << TURN_BIT_POS;
        for (u8 i = 0; i < b.height; ++i)
            res.data |= (b[i] == highest_player) << i;
        return res;
    }

    constexpr bool operator==(gya::compressed_column const &c) const = default;
};

static_assert([] { // loop through all possible columns and verify they are compressed and decompressed properly
    for (usize num_moves = 0; num_moves <= BOARD_HEIGHT; ++num_moves) {
        for (usize i = 0; i < (1ull << num_moves); ++i) {
            gya::board_column c;
            for (usize j = 0; j < num_moves; ++j)
                c.push((i & (1 << j)) ? board::PLAYER_ONE : board::PLAYER_TWO);
            if (c != gya::compressed_column::decompress(gya::compressed_column::compress(c)))
                return false;
        }
    }
    return true;
}());

struct compressed_board {
private:
    std::array<compressed_column, BOARD_WIDTH> data;
public:
    // explicitly default constructors, destructor, and assignment operators, "Rule of 3"
    constexpr compressed_board() = default;

    constexpr compressed_board(gya::compressed_board const &) = default;

    constexpr ~compressed_board() = default;

    constexpr compressed_board &operator=(compressed_board const &) = default;

    // implicit since the conversion is cheap and to simplify usage
    constexpr compressed_board(gya::board const &b) : compressed_board{compress(b)} {}

    // implicit since the conversion is cheap and to simplify usage
    constexpr operator gya::board() const {
        return decompress(*this);
    }

    static constexpr gya::board decompress(gya::compressed_board c) {
        gya::board res;
        for (u8 i = 0; i < BOARD_WIDTH; ++i) {
            res.data[i] = gya::compressed_column::decompress(c.data[i]);
            res.size += res.data[i].height;
        }
        res.winner = res.has_won_test();
        return res;
    }

    static constexpr gya::compressed_board compress(gya::board const &b) {
        gya::compressed_board res;
        for (u8 i = 0; i < BOARD_WIDTH; ++i) {
            res.data[i] = gya::compressed_column::compress(b.data[i]);
        }
        return res;
    }

    constexpr bool operator==(gya::compressed_board const &b) const = default;
};

static_assert([] { // test some randomly generated games to make sure they are compressed and decompressed properly
    constexpr auto NUM_RANDOM_GAMES = 100;
    for (usize i = 0; i < NUM_RANDOM_GAMES; ++i) {
        gya::random_player p1(i), p2(~i);
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
