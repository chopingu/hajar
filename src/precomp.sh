#!/bin/bash
g++ include.hpp -O3 -std=c++20 -fopenmp -march=native -Wpedantic -Wextra -Wall -Wshadow #-Werror
