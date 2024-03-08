#ifndef CUCBET_UTILITIES_CUH
#define CUCBET_UTILITIES_CUH

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cassert>

//--------------------------------------------------
// Aliases
using FPTYPE = double;


//--------------------------------------------------
// Constants
namespace constants {
  // Physical
  inline constexpr FPTYPE   PI { 3.14159265 };     // it's Pi...
  inline constexpr FPTYPE   C0 { 299792458.0 };    // m/s
  inline constexpr FPTYPE EPS0 { 8.854187E-12 };   // F/m
  inline constexpr FPTYPE  MU0 { 1.256637E-6 };    // H/m
  inline constexpr FPTYPE   Me { 9.10938370E-31 }; // electron mass, kg
  inline constexpr FPTYPE   qe { -1.6021766E-19 }; // electron charge, coulombs
  inline constexpr FPTYPE   Kb { 1.38-649E-23 };   // J/K
}

//--------------------------------------------------
// Math Functions
namespace math {
  template<typename T>
  __host__ __device__ T SQR(T x) { return x * x; }

  template<typename T>
  __host__ __device__ T CUBE(T x) { return x * x * x; }

  template <typename T>
  __device__ inline T lerp(T v0, T v1, T t) { return fma(t, v1, fma(-t, v0, v0)); }

  // This function create a uniform spread of points over a given range
  // This should mimic Numpy's linspace function exactly.
  template<typename T>
  std::vector<T> linspace(T start, T stop, size_t n_points, bool endpoint = true) {
    std::vector<T> result(n_points);
    if (endpoint) {
      n_points -= 1;
      result[result.size() - 1] = stop;
    }
    auto delta = (stop - start) / static_cast<T>(n_points);
    T val = start;
    for (size_t i = 0; i < n_points; ++i) {
      result[i] = val;
      val += delta;
    }
    return result;
  }


  // This function creates a set of points that are exactly halfway between the points of the provided vector
  template<typename T>
  std::vector<T> meanspace(const std::vector<T>& input)
  {
    // Initialize the output vector
    std::vector<T> output(input.size()-1);

    // Iterate through, taking the average of each set of points to find the halfway point
    for (size_t i = 0; i < output.size(); i++) {
      output[i] = ( input[i] + input[i+1] ) / 2;
    }

    // Return the staggered vector
    return output;
  }
}

////--------------------------------------------------
//// Matrix Utility Functions
//template<int DIM>
//__global__ void fill_matrix(devMatrix<DIM>& matrix, const FPTYPE value) {
//  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
//  uint32_t stride = blockDim.x * gridDim.x;
//
//  for (uint32_t i = idx; i < matrix.size(); i += stride) {
//    matrix[i] = value;
//  }
//}
//
//template<int DIM>
//__global__ void assert_matrix(devMatrix<DIM>& matrix, const FPTYPE value) {
//  auto i = blockIdx.y;
//  auto j = blockIdx.x;
//  auto k = threadIdx.x;
//
//  if (i < matrix.x && j < matrix.y && k < matrix.z) {
//    assert(matrix(i, j, k) == value);
//  }
//}

#endif //CUCBET_UTILITIES_CUH
