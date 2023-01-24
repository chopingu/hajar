#pragma once

#include "../../include.hpp"

#include "node.hpp"
#include "tree.hpp"

namespace mcts {

class mcts {
public:
    bool m_win;
    bool m_draw;
    i32 m_score;
    u32 m_rollout_limit;

    mcts(u32 rollout_limit) : m_rollout_limit(rollout_limit) {}

    f32 ucb(node *v) {
        if (!v->m_visits) return std::numeric_limits<f32>::max();
        return v->m_score / v->m_visits + 2.f * std::sqrt(std::log(v->m_parent->m_visits) / v->m_visits);
    }

    node *get_child_with_highest_ucb(node *v) {
        f32 mx_ucb = -std::numeric_limits<f32>::max();
        std::vector<node*> mx_children;
        for (auto &child: v->m_children) {
            f32 child_ucb = ucb(child.get());
            if (child_ucb > mx_ucb) {
                mx_ucb = child_ucb;
                mx_children = {child.get()};
            } else if (child_ucb == mx_ucb) {
                mx_children.push_back(child.get());
            }
        }

        return mx_children[std::rand() % mx_children.size()];
    }

    void add_children(node *parent_node, gya::board b, i32 player_id) {
        auto moves = b.get_actions();

        for (auto m: moves) 
            parent_node->m_children.push_back(std::make_unique<node> (node{parent_node, {player_id, m}}));
    }

    void simulate_game(gya::board game, tree *tr, i32 player_id) {
        node *cur_node = tr->m_root.get();
        std::vector<node *> nodes_to_update = {tr->m_root.get()};

        while (!cur_node->is_leaf()) {
            cur_node = get_child_with_highest_ucb(cur_node);
            player_id = cur_node->m_action[0];
            u32 col = static_cast<u32>(cur_node->m_action[1]);
            game.play(col, player_id);
            nodes_to_update.push_back(cur_node);
        }

        gya::game_result result = game.has_won();
        if (!result.is_game_over()) {
            i32 next_player_id = game.turn();
            add_children(cur_node, game, next_player_id);

            game.play(get_child_with_highest_ucb(cur_node)->m_action[1], next_player_id);

            while (1) {
                gya::game_result result2 = game.has_won();
                if (result2.is_game_over()) break;
                next_player_id = game.turn();
                auto moves = game.get_actions();
                game.play(moves[std::rand() % moves.size()], next_player_id);
            }
        }

        result = game.has_won();
        if (result.is_tie()) {
            m_draw = 1;
            m_win = 0;
            m_score = -0.1f;
        } else {
            m_draw = 0;
            m_win = (game.turn() == player_id ? 1 : 0);
            m_score = (m_win ? 1 : -1);
        }

        for (auto node_to_update: nodes_to_update) {
            i32 player_for_node = (node_to_update->m_action)[0];
            i32 node_score = m_score;
            if (player_for_node != player_id)
                node_score *= -1.f;
            (node_to_update->m_visits)++;
            (node_to_update->m_score) += node_score;
        }
    }

    u8 move(gya::board game, i32 player_id) {
        std::unique_ptr<tree> tr = std::make_unique<tree>();

        for (u32 i = 0; i < m_rollout_limit; i++) {
            gya::board copy = game;
            simulate_game(copy, tr.get(), player_id);
        }

        node *mx_child = nullptr;
        i32 mx_visits = -1;
        for (auto &child: tr->m_root->m_children) {
            if (mx_visits < static_cast<i32>(child->m_visits)) {
                mx_visits = child->m_visits;
                mx_child = child.get();
            }
        }

        return (mx_child->m_action)[1];
    }
};

} // namespace mcts
