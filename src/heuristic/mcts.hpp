#pragma once

#include <vector>
#include <memory>
#include <string>

#include "../board.hpp"
#include "../defines.hpp"
#include "../utilities.hpp"

namespace mcts {
template<class T>
class Node {
public:
    u64 visits;
    T value;

    std::shared_ptr<gya::board> state;
    std::vector<std::shared_ptr<Node>> children;
    std::weak_ptr<Node> parent;

    Node(std::weak_ptr<Node> parent, std::shared_ptr<gya::board> state)
            : visits{1}, value{}, state{std::move(state)}, children{}, parent{parent} {}
};
} // namespace mcts
