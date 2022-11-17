cmake_minimum_required(VERSION 3.22)
project(CBET CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(CBET ./src/main.cu src/Utilities/EnumTypes.cuh)

target_include_directories(CBET PUBLIC ./src)

#set_target_properties(CBET PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
