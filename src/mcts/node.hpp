#pragma once

#include "../include.hpp"

namespace mcts {

class node {
public:
    u32 m_visits;
    i32 m_score;

    node *m_parent;
    std::vector<node*> m_children;
    std::array<i32, 2> m_action;

    node(node *parent = nullptr, std::array<i32, 2> action = {}) {
        m_visits = 0;
        m_score = 0;
        m_parent = parent;
        m_action = action;
    }

    bool is_leaf() {
        return m_children.empty();
    }
};

}
