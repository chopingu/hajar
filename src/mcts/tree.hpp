#pragma once

#include "../include.hpp"

#include "node.hpp"

namespace mcts {

class tree {
public:
    node *m_root;

    tree() {
        m_root = new node();
    }
};

} // namespace mcts
