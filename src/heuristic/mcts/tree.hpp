#pragma once

#include "../../include.hpp"

#include "node.hpp"

namespace mcts {

class tree {
public:
    std::unique_ptr<node> m_root;

    tree() : m_root(std::make_unique<node>()) {}
};

} // namespace mcts
