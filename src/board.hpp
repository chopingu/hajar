#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace gya {

    struct board_column {
        std::array<std::uint8_t, 6> data{};
        std::uint8_t height{};

        auto &push(std::uint8_t value) {
            return data[height++] = value;
        }

        auto &operator[](int idx) {
            return data[idx];
        }

        auto &operator[](int idx) const {
            return data[idx];
        }
    };

    struct board {
        std::array<board_column, 7> data{};
        std::int8_t winner = 0;
        std::int8_t size = 0;

        auto &play(std::uint8_t column, std::uint8_t value) {
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
            if (row < 5 && data[col][row + 1] == value) {
                ver++;
                if (row < 4 && data[col][row + 2] == value) {
                    ver++;
                    if (row < 3 && data[col][row + 3] == value) {
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

        [[nodiscard]] int has_won_test() const {
            if (!winner) {
                return size == 42 ? -1 : 0;
            } else {
                return winner;
            }
        }

        [[nodiscard]] int has_won() const {
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
                        data[i][j] == data[i][j + 3])
                        return data[i][j];
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
                        data[i][j] == data[i + 3][j])
                        return data[i][j];
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
                        data[i][j] == data[i + 3][j + 3])
                        return data[i][j];
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
                        data[i][j + 3] == data[i + 3][j])
                        return data[i][j + 3];
                }
            }

            if (std::all_of(data.begin(), data.end(), [](auto &col) { return col.height == 6; }))
                return -1;
            else
                return 0;
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
                ret += std::to_string(j + 1);
                ret += '|';
            }
            ret += '\n';
            return ret;
        }
    };
} // namespace gya
