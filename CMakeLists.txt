cmake_minimum_required(VERSION 3.22)
project(cuCBET CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuCBET ./src/main.cu src/Interpolation/Gradients.cuh)

target_include_directories(cuCBET PUBLIC ./src)

#set_target_properties(CBET PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
