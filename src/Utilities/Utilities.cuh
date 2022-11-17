#ifndef CBET_UTILITIES_CUH
#define CBET_UTILITIES_CUH

//--------------------------------------------------
// C++ Standard Library Includes

// Types
#include <cstdint>
#include <string>
#include <chrono>

// Utilities
#include <cmath>
#include <algorithm>
#include <cassert>

// IO
#include <iostream>
#include <fstream>
#include <iomanip>

//--------------------------------------------------
// Project Includes
#include "Defines.cuh"
#include "BestMatrix.cuh"
#include "Vector.cuh"
#include "EnumTypes.cuh"
#include "Timing.cuh"

//--------------------------------------------------
// Aliases
using devMatrix = matrix_base<float, cuda_managed>;
using vec3 = Vector<float>;

//--------------------------------------------------
// Constants
namespace Constants {
  // Physical
  inline constexpr float   PI { 3.14159265 };     // it's Pi...
  inline constexpr float   C0 { 299792458.0 };    // m/s
  inline constexpr float EPS0 { 8.854187E-12 };   // F/m
  inline constexpr float  MU0 { 1.256637E-6 };    // H/m
  inline constexpr float   Me { 9.10938356E-28 }; // electron mass, kg
  inline constexpr float   qe { 1.6021766E-19 };  // electron charge, coulombs
  inline constexpr float   Kb { 1.38-649E-23 };   // J/K
}

//--------------------------------------------------
// Math Functions
template <typename T>
__host__ __device__ T SQR(T x) { return x * x; }

template<typename T>
__host__ __device__ T CUBE(T x) { return x * x * x; }

//--------------------------------------------------
// Matrix Utility Functions
__global__ void fill_matrix(devMatrix& matrix, const float value) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = idx; i < matrix.size(); i += stride) {
    matrix[i] = value;
  }
}

__global__ void assert_matrix(devMatrix& matrix, const float value) {
  auto i = blockIdx.y;
  auto j = blockIdx.x;
  auto k = threadIdx.x;

  if (i < matrix.x && j < matrix.y && k < matrix.z) {
    assert(matrix(i, j, k) == value);
  }
}

#endif //CBET_UTILITIES_CUH