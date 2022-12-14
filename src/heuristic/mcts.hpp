#pragma once

#include "../include.hpp"

namespace mcts {

const f32 c_puct_base = 19652;
const f32 c_puct_init = 2.5;

template<class T>
class Node {
public:
    i8 m_player = 0;
    i8 m_action = -1;
    u64 m_visits = 0;
    T m_value_sum = 0;
    T m_prior = 0;

    gya::board m_state;
    std::default_random_engine m_gen;
    std::vector<std::unique_ptr<Node>> m_children;

    Node(i8 player, i8 action, T prior)
            : m_player(player), m_action(action), m_prior(prior) {}

    T ucb_score(Node *child) {
        u64 sum = 0; // maybe change for performance reasons
        for (auto ptr: child->m_children) {
            auto grandchild = ptr.get();
            sum += grandchild->m_visits;
        }
        const T c_puct = std::log((child->m_visits + c_puct_base + 1) / c_puct_base) + c_puct_init;
        const T U = c_puct * child->m_prior * std::sqrt(sum) / (child->m_visits + 1);
        const T Q = (child->m_visits) ? -child->m_value_sum / child->m_visits : 0.f;

        return Q + U;
    }

    void expand(const gya::board &state, const std::vector<T> &priors, i8 player) {
        for (u64 i = 0; i < priors.size(); i++) {
            const T prior_prob = priors[i];
            if (prior_prob != 0.0f) {
                auto child = std::make_unique<Node>(prior_prob, -player, i);
                m_children.push_back(std::move(child));
            }
        }
    }

    Node *select_child() {
        T mx = std::numeric_limits<T>::min();
        Node *argmax = m_children.front().get();

        for (auto &ptr: m_children) {
            auto child = ptr.get();
            T score = ucb_score(child);
            if (score > mx) {
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
        for (u64 i = 0; i < m_children.size(); i++) {
            Node *child = m_children[i].get();
            actions.push_back(child->m_action);
            visit_counts.push_back(child->m_visits);

            if (child->m_visits > mx) {
                argmax = i;
                mx = child->m_visits;
            }
        }

        if (!temperature) {
            return actions[argmax];
        } else {
            std::discrete_distribution<u64> distr(visit_counts.begin(), visit_counts.end());
            return actions[distr(m_gen)];
        }
    }
};

// dummy_model until network is finished
template<class T>
std::pair<std::vector<T>, T> dummy_model(gya::board state) {
    std::vector<T> probs(state.data.size());
    for (auto i: state.get_actions())
        probs[i] = 0.5;

    return {probs, 0};
}

/*
template<class T> 
Node *simulation(gya::board m_state, i8 m_player, u8 num_simulations) {
    Node *root = new Node(m_player, -1, 0);
    
    // expand root
    T value;
    std::vector<T> prior_probs;
    tie(prior_probs, value) = dummy_model(m_state);
    root->expand(m_state, prior_probs, m_player);

    for(u64 i = 0; i < num_simulations; i++) {
        Node *cur = root;
        std::vector<Node*> search_path({cur});

        // selection
        while(!cur->m_children.empty()) {
            cur = cur->select_child();
            search_path.push_back(cur);
        }

        Node *parent = search_path[search_path.size()-2];
        m_state = parent->m_state;

        
    }
}
*/

} // namespace mcts
