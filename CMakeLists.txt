cmake_minimum_required(VERSION 3.22)
project(cuCBET CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuCBET ./src/main.cu src/Utilities/PolicyManagers.cuh src/Tests/test_Vector3.cuh src/Beams/Beams2D.cuh src/Beams/Rays2D.cuh src/Utilities/Vector2.cuh)

target_include_directories(cuCBET PUBLIC ./src)

#set_target_properties(CBET PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
