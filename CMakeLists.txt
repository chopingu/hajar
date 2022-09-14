cmake_minimum_required(VERSION 3.22)
project(gya_connect_four)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXE_LINKER_FLAGS "-static")

include_directories(src)

# add_executable(gya_connect_four src/neural_net_testing/test.cpp)
add_executable(gya_connect_four src/main.cpp)
