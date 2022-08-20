cmake_minimum_required(VERSION 3.22)
project(gya_connect_four)

set(CMAKE_CXX_STANDARD 20)

include_directories(src)

add_executable(gya_connect_four src/main.cpp)
