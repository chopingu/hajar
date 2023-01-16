#pragma once

#include "../include.hpp"

namespace heuristic {
struct eval_result {
    bool m_winning: 1;
    bool m_losing: 1;
    i8 m_depth_until_over: 6 = 0;

    constexpr eval_result() : m_winning{}, m_losing{}, m_depth_until_over{} {}

    constexpr eval_result(bool winning, bool losing, i8 depth = 0) : m_winning{winning}, m_losing{losing},
                                                                        m_depth_until_over{depth} {}

    constexpr eval_result incremented() const {
        if (m_winning | m_losing) {
            return eval_result{!m_winning, !m_losing, static_cast<i8>(m_depth_until_over + 1)};
        } else {
            return *this;
        }
    }

    constexpr bool operator>(eval_result other) const {
        if (m_winning && !other.m_winning)
            return true;
        if (!m_losing && other.m_losing)
            return true;
        if (m_winning && other.m_winning)
            return m_depth_until_over < other.m_depth_until_over;
        if (m_losing && other.m_losing)
            return m_depth_until_over > other.m_depth_until_over;
        return false;
    }

    constexpr bool operator<(eval_result other) const {
        return other > *this;
    }

    constexpr bool operator==(eval_result const& other) const = default;

    constexpr operator const char *() const {
        if (is_tied())
            return "TIED";
        else if (is_winning())
            return "WINNING";
        else if (is_losing())
            return "LOSING";
        else
            return "NEUTRAL";
    }

    constexpr bool is_winning() const { return m_winning & !m_losing; }

    constexpr bool is_losing() const { return m_losing & !m_winning; }

    constexpr bool is_tied() const { return m_losing & m_winning; }

    constexpr bool is_game_over() const { return m_losing | m_winning; }
};
static constexpr eval_result WINNING_MOVE{true, false};
static constexpr eval_result LOSING_MOVE{false, true};
static constexpr eval_result NEUTRAL_MOVE{};
static constexpr eval_result TIE_MOVE{true, true};
static_assert(sizeof(eval_result) == sizeof(i8));
}
