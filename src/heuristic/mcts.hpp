#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cmath> 

#include "../board.hpp"
#include "../defines.hpp"
#include "../utilities.hpp"

namespace mcts {

const f32 c_puct_base = 19652;
const f32 c_puct_init = 2.5;

template<class T>
class Node {
public:
    i8 player = 0;
    i8 action = -1;
    u64 visits = 0;
    T value_sum = 0; 
    T prior = 0;

    gya::board state;
    std::default_random_engine gen;
    std::vector<std::unique_ptr<Node>> children;
  
    Node(i8 player, i8 action, T prior) 
        : player(player), action(action), prior(prior) {}

    T ucb_score(Node *child) {
        u64 sum = 0; // maybe change for performance reasons
        for(auto ptr: child->children) {
            auto child = ptr.get();
            sum += child->visits;
        }
        const T c_puct = std::log((child->visits + c_puct_base + 1) / c_puct_base) + c_puct_init;
        const T U = c_puct * child->prior * std::sqrt(sum) / (child->visits + 1);
        const T Q = (child->visits) ? -child->value_sum / child->visits : 0.f; 
        
        return Q + U; 
    }
             
    void expand(const gya::board &state, const std::vector<T> &priors, i8 player) {
        for(u64 i = 0; i < priors.size(); i++) {
            const T prior_prob = priors[i];
            if(prior_prob != 0.0f) { 
                auto child = std::make_unique<Node>(prior_prob, -player, i);
                children.push_back(std::move(child));
            } 
        }
    }

    Node* select_child() {
        T mx = std::numeric_limits<T>::min();
        Node *argmax = children.front().get();
        
        for(auto &ptr: children) {
            auto child = ptr.get();
            T score = ucb_score(child);
            if(score > mx) { 
                mx = score;
                argmax = child;
            }
        }

        return argmax;
    }

    u8 select_action(T temperature) {
        std::vector<u8> actions;
        std::vector<u64> visit_counts;
        
        u8 argmax = 0, mx = 0;
        for(u64 i = 0; i < children.size(); i++) {
            Node *child = children[i].get();
            actions.push_back(child->action);
            visit_counts.push_back(child->visits);

            if(child->visits > mx) {
                argmax = i;
                mx = child->visits;
            }
        }

        if(!temperature) {
            return actions[argmax];
        }
        else {
            std::discrete_distribution<u64> distr(visit_counts.begin(), visit_counts.end());
            return actions[distr(gen)];
        }
    }
};

template<class T>
std::pair<std::vector<T>, T> dummy_model(gya::board state) {
    std::vector<T> probs(state.data.size());
    for(auto i: state.get_actions()) 
        probs[i] = 0.5;

    return {probs, 0};
}

/*
template<class T> 
Node *simulation(gya::board state, i8 player, u8 num_simulations) {
    Node *root = new Node(player, -1, 0);
    
    // expand root
    T value;
    std::vector<T> prior_probs;
    tie(prior_probs, value) = dummy_model(state);
    root->expand(state, prior_probs, player);

    for(u64 i = 0; i < num_simulations; i++) {
        Node *cur = root;
        std::vector<Node*> search_path({cur});

        // selection
        while(!cur->children.empty()) {
            cur = cur->select_child();
            search_path.push_back(cur);
        }

        Node *parent = search_path[search_path.size()-2];
        state = parent->state;

        
    }
}
*/

} // namespace mcts
