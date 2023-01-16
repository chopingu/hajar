#pragma once

#include "../include.hpp"

#include <unistd.h>

#include "layer/create_layer.hpp"
#include "layer/layer_base.hpp"
#include "layer/input_layer.hpp"
#include "layer/fully_connected_layer.hpp"

#include "utils/activation.hpp"
#include "utils/cost.hpp"
#include "utils/math.hpp"
#include "utils/tensor.hpp"
#include "utils/solver.hpp"

namespace pinguml {

const u8 BATCH_RESERVED = 1, BATCH_FREE = 0, BATCH_COMPLETE = 2;
const i8 BATCH_FILLED_COMPLETE = -2, BATCH_FILLED_IN_PROCESS = -1;
const u8 MAIN_COPY = 0;

class network {
private:

#if defined(OMP)
    omp_lock_t m_lock_batch;
    void lock_batch() { omp_set_lock(&m_lock_batch); }
    void unlock_batch() { omp_unset_lock(&m_lock_batch); }
    void init_lock() { omp_init_lock(&m_lock_batch); }
    void destroy_lock() { omp_destroy_lock(&m_lock_batch); }
    u32 thread_num() { return omp_get_thread_num(); }
#else
    void lock_batch() {}
    void unlock_batch() {}
    void init_lock() {}
    void destroy_lock() {}
    u32 thread_num() { return 0; }
#endif

    u32 m_output_size;
    u32 m_nr_threads;
    u32 m_batch_size;  

    cost_base *m_cost_f;
    solver_base *m_solver;

public:	
    u32 m_current_epoch;
    u32 m_nr_epochs;

    std::unordered_map<std::string, u32> m_layer_id;  
    std::vector<std::pair<std::string, std::string>> m_edge_list;
    std::vector<tensor*> m_connections; 

    std::vector<std::vector<layer_base*>> m_network_copies;
    std::vector<std::vector<tensor>> m_delta_weights_copies; 
    std::vector<std::vector<tensor>> m_delta_biases_copies; 
    std::vector<u8> m_batch_statuses;


    network(std::string solver = nullptr): m_output_size(0), m_nr_threads(1) { 
        m_batch_size = 1;
        m_cost_f = nullptr;
        m_solver = create_solver(solver);
        m_network_copies.resize(1);
        m_delta_weights_copies.resize(m_batch_size);
        m_delta_biases_copies.resize(m_batch_size);
        m_batch_statuses.resize(m_batch_size);
        m_current_epoch = 0; 
        m_nr_epochs = 1000;
    }

    ~network() { destroy_lock(); }

    u32 output_size() { return m_output_size; }

    u32 nr_thread() { return m_nr_threads; }

    void sync_network_copies() {
        for(u32 i = 1; i < (u32)m_network_copies.size(); i++)
            for(u32 j = 0; j < (u32)m_network_copies[MAIN_COPY].size(); j++)
                for(u32 k = 0; k < m_network_copies[MAIN_COPY][j]->m_biases.size(); k++) 
                    m_network_copies[i][j]->m_biases.m_ptr[k] = m_network_copies[MAIN_COPY][j]->m_biases.m_ptr[k];
    }

    void create_network_copies() {
        u32 cur_nr_batches = (u32)m_network_copies.size();
        if (cur_nr_batches < m_nr_threads) m_network_copies.resize(m_nr_threads);
        // (cur_nr_bathces > m_nr_threads) ?
        sync_network_copies();
    }

    void enable_threads(u32 nr_threads = 0) {
#if defined(OMP)
        if (!nr_threads) nr_threads = omp_get_num_procs();
        m_nr_threads = nr_threads;
        omp_set_nested(1);
        omp_set_num_threads(m_nr_threads);
#else
        if (nr_threads > 1) throw std::runtime_error("Define OpenMP for threading");
        m_nr_threads = 1;
#endif
        create_network_copies();
    }

    bool push_back(const std::string name, const std::string build) {
        if(m_layer_id[name]) return false; 

        create_network_copies();

        m_layer_id[name] = (u32)m_network_copies[MAIN_COPY].size();

        layer_base *layer = create_layer(name, build);
        m_network_copies[MAIN_COPY].push_back(layer);

        m_output_size = layer->m_nodes.m_rows * layer->m_nodes.m_cols * layer->m_nodes.m_channels;

        for(u32 i = 1; i < (u32)m_network_copies.size(); i++) 
            m_network_copies[i].push_back(create_layer(name, build));

        return true;
    }

    void connect(const std::string left_name, const std::string right_name) {
        m_edge_list.push_back(std::make_pair(left_name, right_name));

        const u32 left_index = m_layer_id[left_name];
        const u32 right_index = m_layer_id[right_name];

        layer_base *left_layer = m_network_copies[MAIN_COPY][left_index];
        layer_base *right_layer = m_network_copies[MAIN_COPY][right_index];

        u32 connection_index = (u32)m_connections.size();
        tensor *connection = right_layer->create_connection(*left_layer, connection_index);
        m_connections.push_back(connection);

        for(u32 i = 1; i < (u32)m_network_copies.size(); i++) {
            left_layer = m_network_copies[i][left_index];
            right_layer = m_network_copies[i][right_index];
            delete right_layer->create_connection(*left_layer, connection_index);
        }

        if (m_solver) {
            if(connection) m_solver->push_back(connection->m_rows, connection->m_cols, connection->m_channels);
            else m_solver->push_back(1, 1, 1);
        }

        if(connection) connection->fill(1);
    }

    void connect() {	
        for(u32 i = 0; i < (u32)m_network_copies[MAIN_COPY].size() - 1; i++) 
            connect(m_network_copies[MAIN_COPY][i]->m_name, m_network_copies[MAIN_COPY][i + 1]->m_name);
    }

    i32 layer_index(const std::string name) {
        if(m_layer_id.find(name) != m_layer_id.end()) return m_layer_id[name];
        return -1;
    }

    f32 *forward(const f32 *input, i32 copy_id = -1, u32 training=0) {
        if(copy_id < 0) copy_id = thread_num();
        if (copy_id > (i32)m_nr_threads && m_nr_threads > 0)
            throw std::runtime_error("threading error\n");
        if (copy_id >= (i32)m_network_copies.size())
            throw std::runtime_error("threading error\n");

        std::vector<layer_base*> inputs;
        for(auto layer: m_network_copies[copy_id]) {
            if (dynamic_cast<input_layer*>(layer) != nullptr) inputs.push_back(layer);
            layer->m_nodes.fill(0.f);
        }

        const f32 *input_copy = input;

        for(auto layer: inputs) {
            std::memcpy(layer->m_nodes.m_ptr, input_copy, sizeof(f32) * layer->m_nodes.size());
            input_copy += layer->m_nodes.size();
        }

        for(auto layer: m_network_copies[copy_id]) {
            layer->activate(); 

            for(auto &i: layer->m_forward_connections) {
                u32 connection_index = i.first; 
                layer_base *right_layer = i.second;

                right_layer->push_forward(*layer, *m_connections[connection_index], training);
            }

        }

        return m_network_copies[copy_id][m_network_copies[copy_id].size()-1]->m_nodes.m_ptr;
    }

    u32 arg_max(const f32 *output, const u32 size) {
        u32 index = 0;
        for (u32 i = 0; i < size; i++) 
            if (output[index] < output[i])  
                index = i;
            
        return index;
    }

    u32 predict_class(const f32 *input, i32 copy_id = -1) {
        const f32* output = forward(input, copy_id);
        return arg_max(output, m_output_size);
    }

    u32 batch_size() { return m_batch_size; }

    void reset_mini_batch() { 
        m_batch_statuses.assign(m_batch_statuses.size(), BATCH_FREE);
    }

    void set_batch_size(u32 batch_size) {
        m_batch_size = std::max(batch_size, (u32)1);
        m_delta_weights_copies.resize(m_batch_size);
        m_delta_biases_copies.resize(m_batch_size);
        m_batch_statuses.resize(m_batch_size); 
        reset_mini_batch();
    }

    int next_free_batch() {
        u32 reserved = 0;
        u32 filled = 0;
        for (u32 i = 0; i < (u32)m_batch_statuses.size(); i++) {
            if (m_batch_statuses[i] == BATCH_FREE) return i;
            if (m_batch_statuses[i] == BATCH_RESERVED) reserved++;
            if (m_batch_statuses[i] == BATCH_COMPLETE) filled++;
        }
        if (reserved > 0) return BATCH_FILLED_IN_PROCESS; 
        if (filled == m_batch_statuses.size()) return BATCH_FILLED_COMPLETE;

        throw std::runtime_error("threading error"); 
    }

    void sync_mini_batch() {
        i8 next = next_free_batch();
        if (next == BATCH_FILLED_IN_PROCESS) throw std::runtime_error("thread locked");

        u32 nr_layers = (u32)m_network_copies[MAIN_COPY].size();

        // sum contributions in delta weights and delta biases
        layer_base *layer;
        for (i32 i = (i32)nr_layers - 1; i >= 0; i--) {
            layer = m_network_copies[MAIN_COPY][i];

            for(auto &j: layer->m_backward_connections) {
                u32 connection_index = (u32)j.first;

                if (m_batch_statuses[MAIN_COPY] == BATCH_FREE) 
                    m_delta_weights_copies[MAIN_COPY][connection_index].fill(0);

                for (u32 k = 1; k < m_batch_size; k++) 
                    if (m_batch_statuses[k] == BATCH_COMPLETE) 
                        m_delta_weights_copies[MAIN_COPY][connection_index] += m_delta_weights_copies[k][connection_index];
            }

            //if (dynamic_cast<convolution_layer*> (layer) != NULL)  continue;

            if (m_batch_statuses[MAIN_COPY] == BATCH_FREE) 
                m_delta_biases_copies[MAIN_COPY][i].fill(0);

            for (u32 j = 1; j < m_batch_size; j++) 
                if (m_batch_statuses[j] == BATCH_COMPLETE) 
                    m_delta_biases_copies[MAIN_COPY][i] += m_delta_biases_copies[j][i];
        }

        // update weights and add delta biases to main copy 
        for (i32 i = nr_layers - 1; i >= 0; i--) {
            layer = m_network_copies[MAIN_COPY][i];

            for(auto &j: layer->m_backward_connections) {
                u32 connection_index = (u32)j.first;
                if (m_delta_weights_copies[MAIN_COPY][connection_index].size())
                    if(m_connections[connection_index]) 
                        m_solver->update_weights(m_connections[connection_index], connection_index, m_delta_weights_copies[MAIN_COPY][connection_index]);

            }

            layer->update_biases(m_delta_biases_copies[MAIN_COPY][i], m_solver->m_learning_rate);
        }

        reset_mini_batch();
        sync_network_copies();
    }

    i32 reserve_next_batch() {
        lock_batch();
        i32 index = -3;
        while (index < 0) {
            index = next_free_batch();
            if (index >= 0) { 
                m_batch_statuses[index] = BATCH_RESERVED;
                unlock_batch();
                return index;
            }
            else if (index == BATCH_FILLED_COMPLETE) { 
                sync_mini_batch(); 
                index = next_free_batch();
                m_batch_statuses[index] = BATCH_RESERVED;
                unlock_batch();
                return index;
            }
            unlock_batch();
            usleep(1 * 1000);
            lock_batch();
        }
        return -3;
    }

    f32 learning_rate() {
        if(!m_solver) throw std::runtime_error("set solver"); 
        return m_solver->m_learning_rate;
    }

    void set_learning_rate(f32 alpha) {
        if(!m_solver) throw std::runtime_error("set solver"); 
        m_solver->m_learning_rate = alpha;
    }

    void reset_solver() {
        if(!m_solver) throw std::runtime_error("set solver"); 
        m_solver->reset();
    }

    void set_nr_epochs(u32 mx) { 
        m_nr_epochs = std::max(mx, (u32)1);
    }

    u32 current_epoch() { 
        return m_current_epoch;
    }

    void start_epoch(std::string cost_function = "mse") {
        m_cost_f = create_cost(cost_function);

        if (!m_current_epoch) reset_solver();
    }

    bool over() {
        if (m_current_epoch > m_nr_epochs) return true;
        else return false;
    }

    bool end_epoch() {
        sync_mini_batch();
        
        m_current_epoch++;

        return over();
    }

    void backpropogation(const u32 batch_index, const u32 copy_id) {
        const u32 nr_layers = (u32)m_network_copies[copy_id].size();

        layer_base *layer;
        for (i32 i = nr_layers - 1; i >= 0; i--) {
            layer = m_network_copies[copy_id][i];
            u32 layer_size = layer->m_nodes.size();

            if (i < (i32)nr_layers - 1)
                for (u32 j = 0; j < layer_size; j++)
                    layer->m_delta.m_ptr[j] *= layer->df(layer->m_nodes.m_ptr, (u32)j);

            for(auto &j: layer->m_backward_connections) {
                layer_base *left_layer = j.second;
                layer->propogate_delta(*left_layer, *m_connections[j.first]);
            }
        }

        u32 nr_connections = (u32)m_connections.size();
        m_delta_weights_copies[batch_index].resize(nr_connections);
        m_delta_biases_copies[batch_index].resize(nr_layers);

        for (i32 i = nr_layers - 1; i >= 0; i--) {
            layer = m_network_copies[copy_id][i];

            for(auto &j: layer->m_backward_connections) {
                layer_base *left_layer = j.second;
                u32 connection_index = (u32)j.first;
                
                //if (dynamic_cast<max_pooling_layer*> (layer) != NULL)  continue;
                
                layer->calculate_delta_weights(*left_layer, m_delta_weights_copies[batch_index][connection_index]);
            }
            //if (dynamic_cast<convolution_layer*> (layer) != NULL)  continue;

            m_delta_biases_copies[batch_index][i] = layer->m_delta;
        }

        lock_batch();

        m_batch_statuses[batch_index] = BATCH_COMPLETE;
        i32 next_index = next_free_batch();
        if (next_index == BATCH_FILLED_COMPLETE) 
            sync_mini_batch(); 
                               
        unlock_batch();
    }

    tensor make_input(f32 *input, const u32 copy_id) { 
        tensor in;

        u32 size = 0;
        std::vector<layer_base*> inputs;
        for(auto layer: m_network_copies[copy_id]) {
            if (dynamic_cast<input_layer*> (layer) != nullptr) {
                inputs.push_back(layer);
                size += layer->m_nodes.size();
            }
        }

        in.resize(size, 1, 1);
        std::memcpy(in.m_ptr, input, sizeof(f32) * size);
            
        return in;
    }

    bool train_class(f32 *in, u32 label_index, i32 copy_id = -1) {
        if (m_solver == nullptr) throw std::runtime_error("set solver");
        if (copy_id < 0) copy_id = thread_num();
        if (copy_id > (i32)m_nr_threads) throw std::runtime_error("out of bounds thread");

        f32 *input = in;

        i32 batch_index = reserve_next_batch();
        if (batch_index < 0) return false;

        forward(input, copy_id, 1);
        
        for(auto layer: m_network_copies[copy_id]) layer->m_delta.fill(0.f);

        u32 nr_layers = (u32)m_network_copies[copy_id].size();

        layer_base *layer = m_network_copies[copy_id][nr_layers - 1];
        const u32 size = layer->m_nodes.size();

        //if (dynamic_cast<dropout_layer*> (layer) != NULL) bail("can't have dropout on last layer");

        std::vector<f32> target;
        if(layer->m_f->m_name == "sigmoid" || layer->m_f->m_name == "softmax" || layer->m_f->m_name == "brokemax")
            target = std::vector<f32>(size, 0.f);
        else
            target = std::vector<f32>(size, -1.f);

        if(label_index < size) target[label_index] = 1;

        f32 cost_activation_type = 0;
        if (layer->m_f->m_name == "sigmoid" && m_cost_f->m_name == "cross_entropy") cost_activation_type = 1;
        else if (layer->m_f->m_name == "softmax" && m_cost_f->m_name == "cross_entropy") cost_activation_type = 1;
        else if (layer->m_f->m_name == "tanh" && m_cost_f->m_name == "cross_entropy") cost_activation_type = 4;

        for (u32 i = 0; i < size; i++) {
            if(cost_activation_type)
                layer->m_delta.m_ptr[i] = cost_activation_type * (layer->m_nodes.m_ptr[i] - target[i]);
            else
                layer->m_delta.m_ptr[i] = m_cost_f->cost_d(layer->m_nodes.m_ptr[i], target[i]) * layer->df(layer->m_nodes.m_ptr, i);
        }

        backpropogation(batch_index, copy_id);

        return true;
    }

    // if positive=1, goal is to minimize the distance between in and target
    bool train_target(f32 *in, f32 *target, i32 positive = 1, i32 copy_id = -1) {
        if (m_solver == NULL) throw std::runtime_error("set solver");
        if (copy_id < 0) copy_id = thread_num();
        if (copy_id > (i32)m_nr_threads) throw std::runtime_error("need to enable OMP");

        tensor input = make_input(in, copy_id);

        i32 batch_index = reserve_next_batch();
        if (batch_index < 0) return false;

        forward(in, copy_id, 1);

        for(auto layer: m_network_copies[copy_id]) layer->m_delta.fill(0.f);

        u32 nr_layers = (u32)m_network_copies[copy_id].size();

        layer_base *layer = m_network_copies[copy_id][nr_layers - 1];
        const u32 size = layer->m_nodes.size();

        //if (dynamic_cast<dropout_layer*> (layer) != NULL) bail("can't have dropout on last layer");

        f32 cost_activation_type = 0;
        if (layer->m_f->m_name == "sigmoid" && m_cost_f->m_name == "cross_entropy") cost_activation_type = 1;
        else if (layer->m_f->m_name == "softmax" && m_cost_f->m_name == "cross_entropy") cost_activation_type = 1;
        else if (layer->m_f->m_name == "tanh" && m_cost_f->m_name == "cross_entropy") cost_activation_type = 4;

        for (u32 i = 0; i < size; i++) {
            if (positive) { 
                if (cost_activation_type > 0)
                    layer->m_delta.m_ptr[i] = cost_activation_type * (layer->m_nodes.m_ptr[i] - target[i]);
                else
                    layer->m_delta.m_ptr[i] = m_cost_f->cost_d(layer->m_nodes.m_ptr[i], target[i]) * layer->df(layer->m_nodes.m_ptr, i);
            }
            else {
                if (cost_activation_type > 0)
                    layer->m_delta.m_ptr[i] = cost_activation_type * (1.f - abs(layer->m_nodes.m_ptr[i] - target[i]));
                else
                    layer->m_delta.m_ptr[i] = (1.f - abs(m_cost_f->cost_d(layer->m_nodes.m_ptr[i], target[i]))) * layer->df(layer->m_nodes.m_ptr, i);
            }
        }

        backpropogation(batch_index, copy_id);

        return true;
    }

    std::string configuration() {
        std::string str;
        for (u32 i = 0; i < (u32)m_network_copies[MAIN_COPY].size(); i++) 
            str += "  " + std::to_string(i) + " : " + m_network_copies[MAIN_COPY][i]->m_name + " : " + m_network_copies[MAIN_COPY][i]->config_string();

        str += "\n";

        if (!m_edge_list.size()) return str;

        for (u32 i = 0; i < (u32)m_edge_list.size(); i++) {
            if (i % 3 == 0) str += "  ";
            if((i % 3 == 1) || (i % 3 == 2)) str += ", ";
            str += m_edge_list[i].first + "-" + m_edge_list[i].second;
            if (i % 3 == 2) str += "\n";
        }

        return str;
    }

    bool write(std::ofstream& ofs, [[maybe_unused]] bool final = 0) {
        u32 layer_cnt = (u32)m_network_copies[MAIN_COPY].size();

        ofs << "pinguml" << std::endl;
        ofs << (u32)(layer_cnt) << std::endl;

        for(u32 j = 0; j < (u32)m_network_copies[0].size(); j++)
            ofs << m_network_copies[MAIN_COPY][j]->m_name << std::endl << m_network_copies[MAIN_COPY][j]->config_string();

        //			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)

        ofs << (u32)m_edge_list.size() << std::endl;
        for(u32 j = 0; j < (u32)m_edge_list.size(); j++)
            ofs << m_edge_list[j].first << std::endl << m_edge_list[j].second << std::endl;

        ofs << (u32)0 << std::endl;

        for(u32 j = 0; j < (u32)m_network_copies[MAIN_COPY].size(); j++) {
            if (m_network_copies[MAIN_COPY][j]->uses_biases()) {
                for (u32 k = 0; k < m_network_copies[MAIN_COPY][j]->m_biases.size(); k++) ofs << m_network_copies[MAIN_COPY][j]->m_biases.m_ptr[k] << " ";
                ofs << std::endl;
            }
        }

        for(u32 j = 0; j < (u32)m_connections.size(); j++) {
            if (m_connections[j]) {
                for (u32 i = 0; i < m_connections[j]->size(); i++) ofs << m_connections[j]->m_ptr[i] << " ";
                ofs << std::endl;
            }
        }
        
        ofs.flush();

        return true;
    }

    bool write(std::string &filename, bool final = 0) { 
        std::ofstream temp((const char*)filename.c_str(), std::ios::binary);
        return write(temp, final);
    }


    std::string getcleanline(std::istream& ifs) {
        std::string s;

        std::istream::sentry se(ifs, true);
        std::streambuf* sb = ifs.rdbuf();

        for (;;) {
            int c = sb->sbumpc();
            switch (c) {
                case '\n':
                    return s;
                case '\r':
                    if (sb->sgetc() == '\n') sb->sbumpc();
                    return s;
                case EOF:
                    if (s.empty()) ifs.setstate(std::ios::eofbit);
                    return s;
                default:
                    s += (char)c;
            }
        }
    }

    bool read(std::istream &ifs) {
        if(!ifs.good()) return false;
        std::string s;
        s = getcleanline(ifs);
        u32 nr_layers;
        if (s == "pinguml") {
            s = getcleanline(ifs);
            nr_layers = stoi(s);
        }
        else if (s == "pinguml:") {
            u32 cnt = 1;

            while (!ifs.eof()) {
                s = getcleanline(ifs);
                if (s.empty()) continue;
                if(s[0] == '#') continue;
                push_back(std::to_string(cnt), s);
                cnt++;
            }

            connect();

            sync_network_copies();

            return true;
        }
        else
            nr_layers = std::stoi(s);

        std::string layer_name;
        std::string layer_build;
        for (u32 i = 0; i < (u32)nr_layers; i++) {
            layer_name = getcleanline(ifs);
            layer_build = getcleanline(ifs);
            push_back(layer_name, layer_build);
        }

        i32 graph_count;
        ifs >> graph_count;
        getline(ifs, s); 
        if (graph_count <= 0) {
            connect();
        }
        else {
            std::string layer_name1;
            std::string layer_name2;
            for (u32 i = 0; i < (u32)graph_count; i++) {
                layer_name1 = getcleanline(ifs);
                layer_name2 = getcleanline(ifs);
                connect(layer_name1, layer_name2);
            }
        }

        u32 binary;
        s = getcleanline(ifs); 
        binary = std::stoi(s);

        if(binary == 1) {
            for(u32 i = 0; i < (u32)m_network_copies[MAIN_COPY].size(); i++)
                if (m_network_copies[MAIN_COPY][i]->uses_biases()) 
                    ifs.read((char*)m_network_copies[MAIN_COPY][i]->m_biases.m_ptr, m_network_copies[MAIN_COPY][i]->m_biases.size() * sizeof(f32));

            for (u32 i = 0; i < (u32)m_connections.size(); i++) 
                if (m_connections[i]) 
                    ifs.read((char*)m_connections[i]->m_ptr, m_connections[i]->size() * sizeof(f32));
            
        }
        else if(binary == 0) {
            for(u32 i = 0; i < nr_layers; i++) {
                if (m_network_copies[MAIN_COPY][i]->uses_biases()) {
                    for (u32 j = 0; j < m_network_copies[MAIN_COPY][i]->m_biases.size(); j++) 
                        ifs >> m_network_copies[MAIN_COPY][i]->m_biases.m_ptr[j];

                    ifs.ignore();
                }
            }

            for (u32 i = 0; i < (u32)m_connections.size(); i++) {
                if (m_connections[i]) {
                    for (u32 j = 0; j < m_connections[i]->size(); j++) 
                        ifs >> m_connections[i]->m_ptr[j];

                    ifs.ignore(); 
                }
            }
        }

        sync_network_copies();

        return true;
    }

    bool read(std::string filename) {
        std::ifstream fs(filename.c_str(), std::ios::binary);
        if (fs.is_open()) {
            bool ret = read(fs);
            fs.close();
            return ret;
        }
        else return false;
    }
};

} // namespace pinguml
