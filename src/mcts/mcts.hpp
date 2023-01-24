#pragma once 

#include "../include.hpp"

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
        if(!(v->m_visits)) return 1e9;
        return (v->m_score) / (v->m_visits + 2.f * std::sqrt(std::log(v->m_parent->m_visits) / (v->m_visits)));
    }

    node *get_child_with_highest_ucb(node *v) {
        f32 mx_ucb = -1e9;
        std::vector<node*> mx_children;
        for(auto child: v->m_children) {
            f32 child_ucb = ucb(child);
            if(child_ucb > mx_ucb) {
                mx_ucb = child_ucb;
                mx_children = {child};
            }
            else if(child_ucb == mx_ucb) {
                mx_children.push_back(child);
            }
        }

        return mx_children[std::rand()%(mx_children.size())];
    }

    void add_children(node *parent_node, gya::board b, i32 player_id) {
        auto moves = b.get_actions();

        std::vector<node*> children;
        for(auto m: moves) {
            node *child_node = new node(parent_node, {player_id, m});
            children.push_back(child_node);
        }

        parent_node->m_children = children;
    }

    void simulate_game(gya::board game, tree *tr, i32 player_id) {
        node *cur_node = tr->m_root;
        std::vector<node*> nodes_to_update = {cur_node};

        while(!(cur_node->is_leaf())) {
            cur_node = get_child_with_highest_ucb(cur_node);
            player_id = (cur_node->m_action)[0];
            u32 col = (u32)(cur_node->m_action)[1];
            game.play(col, player_id);
            nodes_to_update.push_back(cur_node);
        }

        gya::game_result result = game.has_won();
        if(!result.is_game_over()) {
            i32 next_player_id = game.turn();
            add_children(cur_node, game, next_player_id);

            node *best_new_child_node = get_child_with_highest_ucb(cur_node);
            game.play((best_new_child_node->m_action)[1], next_player_id);

            while(1) {
                gya::game_result result2 = game.has_won(); 
                if(result2.is_game_over()) break;
                next_player_id = game.turn();
                auto moves = game.get_actions();
                game.play(moves[std::rand()%(moves.size())], next_player_id);
            }
        }

        result = game.has_won();
        if(result.is_tie()) {
            m_draw = 1;
            m_win = 0;
            m_score = 0;
        }
        else if(result.player_1_won()) {
            m_draw = 0;
            m_score = m_win = (game.turn() == gya::board::PLAYER_ONE ? 1 : -1);
        }
        else {
            m_draw = 0;
            m_score = m_win = (game.turn() == gya::board::PLAYER_TWO ? 1 : -1);
        }

        for(auto node_to_update: nodes_to_update) {
            i32 player_for_node = (node_to_update->action)[0];
            i32 node_score = m_score;
            if(player_for_node != player_id) 
                node_score *= -1.f;
            (node_to_update->m_visits)++;
            (node_to_update->m_score) += node_score;
        }
    }

    u32 play(gya::board game, i32 player_id) {
        tree *tr = new tree();

        for(u32 i = 0; i < m_rollout_limit; i++) {
            gya::board copy = game;
            simulate_game(copy, tr, player_id);
        }

        node *mx_child;
        u32 mx_visits = -1;
        for(auto child: tr->m_root->m_children) 
            if(mx_visits < child->m_visits) {
                mx_visits = child->m_visits;
                mx_child = child;
            }

        return (mx_child->m_action)[1];
    }
};

}
