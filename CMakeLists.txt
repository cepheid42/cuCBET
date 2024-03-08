cmake_minimum_required(VERSION 3.22)
project(cuCBET CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(cuCBET ./main.cu)

target_include_directories(cuCBET PUBLIC ./include)
