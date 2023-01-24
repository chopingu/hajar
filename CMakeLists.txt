cmake_minimum_required(VERSION 3.20)
project(gya_connect_four)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Werror -Wpedantic -Wextra -Wall -Wshadow -march=native")
set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(gya_connect_four src/main.cpp)

include_directories(src)
