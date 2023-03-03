#ifndef CUCBET_UTILITIES_CUH
#define CUCBET_UTILITIES_CUH

//--------------------------------------------------
// C++ Standard Library Includes

// Types
//#include <vector>
#include <cstdint>
//#include <string>
//#include <chrono>

// Utilities
#include <cmath>
//#include <algorithm>
#include <cassert>

// IO
#include <iostream>
#include <fstream>
#include <iomanip>
//#include <sstream>

//--------------------------------------------------
// Project Includes
#include "BesterMatrix.cuh"
#include "Vector2.cuh"
#include "Timing.cuh"

//--------------------------------------------------
// Aliases
using FPTYPE = float;
template<int DIM> using devMatrix = matrix_base<FPTYPE, DIM, cuda_managed>;

//--------------------------------------------------
// Constants
namespace Constants {
  // Physical
  inline constexpr FPTYPE   PI { 3.14159265 };     // it's Pi...
  inline constexpr FPTYPE   C0 { 299792458.0 };    // m/s
  inline constexpr FPTYPE EPS0 { 8.854187E-12 };   // F/m
  inline constexpr FPTYPE  MU0 { 1.256637E-6 };    // H/m
  inline constexpr FPTYPE   Me { 9.10938356E-28 }; // electron mass, kg
  inline constexpr FPTYPE   qe { 1.6021766E-19 };  // electron charge, coulombs
  inline constexpr FPTYPE   Kb { 1.38-649E-23 };   // J/K
}

//--------------------------------------------------
// Math Functions
template <typename T>
__host__ __device__ T SQR(T x) { return x * x; }

template<typename T>
__host__ __device__ T CUBE(T x) { return x * x * x; }

//--------------------------------------------------
// Matrix Utility Functions
template<int DIM>
__global__ void fill_matrix(devMatrix<DIM>& matrix, const FPTYPE value) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = idx; i < matrix.size(); i += stride) {
    matrix[i] = value;
  }
}

template<int DIM>
__global__ void assert_matrix(devMatrix<DIM>& matrix, const FPTYPE value) {
  auto i = blockIdx.y;
  auto j = blockIdx.x;
  auto k = threadIdx.x;

  if (i < matrix.x && j < matrix.y && k < matrix.z) {
    assert(matrix(i, j, k) == value);
  }
}

#endif //CUCBET_UTILITIES_CUH
