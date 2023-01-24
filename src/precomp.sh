#!/bin/bash
g++ include.hpp -O3 -O3 -std=c++20 -fopenmp -march=native -isystem ../lib/tiny_dnn/ -Wpedantic -Wextra -Wall -Wshadow #-Werror
