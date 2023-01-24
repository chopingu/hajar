#!/bin/bash
time g++ $1 -O3 -std=c++20 -fopenmp -march=native -isystem ../lib/tiny_dnn/ -Wpedantic -Wextra -Wall -Wshadow -Werror -Wfatal-errors
