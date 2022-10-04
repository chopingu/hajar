#pragma once

#include <array>
#include <string>
#include <string_view>
#include <algorithm>
#include <cassert>
#include <iostream>

#include "defines.hpp"

namespace gya {
struct game_result {
    i8 state{};

    constexpr void set_player_1_won() noexcept {
        state = 1;
    }

    constexpr void set_player_2_won() noexcept {
        state = 2;
    }

    constexpr void set_tie() noexcept {
        state = -1;
    }

    constexpr int winner() const noexcept { return state; }

    constexpr bool player_1_won() const { return state == 1; }

    constexpr bool player_2_won() const { return state == 2; }

    constexpr bool is_game_over() const { return player_1_won() || player_2_won() || is_tie(); }

    constexpr bool is_tie() const { return state == -1; }
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
        return data.at(height++) = value;
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
    i8 winner = 0;
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

    constexpr i8 &play(u8 column, i8 value = 10) {
        if (value == 10)
            value = turn();
        if (value != 1 && value != -1) {
            std::cout << std::flush;
            std::cerr << "invalid player value\n";
            exit(-1);
        }
        if (data[column].height >= 6) {
            std::cout << std::flush;
            std::cerr << std::flush;
            std::cerr << to_string() << std::endl;
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
            winner = value;
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
            winner = value;
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
            winner = value;
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
            winner = value;
            return ret;
        }

        return ret;
    }

    std::vector<u8> get_actions() {
        std::vector<u8> res;
        for (u8 i = 0; i < 7; ++i)
            if (data[i].height < 6)
                res.push_back(i);
        return res;
    }

    [[nodiscard]] constexpr game_result has_won() const {
        return winner != 0 ?
               winner == 1 ? game_result{1} : game_result{2} :
               size == 42 ? game_result{-1} : game_result{0};
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

    constexpr i8 turn() const {
        return size % 2 ? -1 : 1;
    }
};
} // namespace gya
