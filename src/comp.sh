#!/bin/bash
time g++ $1 -O3 -std=c++20 -fopenmp -march=native -Wpedantic -Wextra -Wall -Wshadow -Werror
