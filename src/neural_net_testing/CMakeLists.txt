cmake_minimum_required(VERSION 3.22)
project(gya_testing)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXE_LINKER_FLAGS "-static")

include_directories(../gya/src)
include_directories(../lmj-misc/src)
add_executable(gya_testing main.cpp)
