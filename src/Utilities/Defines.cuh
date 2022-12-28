#ifndef CBET_DEFINES_CUH
#define CBET_DEFINES_CUH

#include <stdexcept>
#include <string>
#include <cstdint>
#include <iostream>

#define _hd __host__ __device__

#define logAssert(condition, message)                         \
if (!condition) {                                             \
  throw std::runtime_error(std::string(__FILE__) + "(" +      \
                           std::string(__FUNCTION__) + "," +  \
                           std::to_string(__LINE__) + "): " + \
                           message + "\n");                   \
}

#define cudaChk(ans)                                          \
if (ans != cudaSuccess) {                                     \
  throw std::runtime_error(std::string(__FILE__) + "(" +      \
                          std::string(__FUNCTION__) + "," +   \
                          std::to_string(__LINE__) + "): " +  \
                          "GPU Error: " +                     \
                          cudaGetErrorString(ans) + "\n");    \
}

#define assert_in_bounds(arr, val)                             \
if (val >= arr.size()) {                                       \
  throw std::runtime_error(std::string(__FILE__) + "(" +       \
                           std::string(__FUNCTION__) + "," +   \
                           std::to_string(__LINE__) + "): " +  \
                           " Out of bounds index: " +          \
                           std::to_string(val));               \
}

#endif //CBET_DEFINES_CUH
