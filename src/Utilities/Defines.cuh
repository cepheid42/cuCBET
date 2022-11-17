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

#define cudaChk(ans) { gpuAssert((ans)); }
inline void gpuAssert(cudaError_t code) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string(__FILE__) + "(" +
                             std::string(__FUNCTION__) + "," +
                             std::to_string(__LINE__) + "): " + "GPU Error: " +
                             cudaGetErrorString(code) + "\n");
  }
}
#endif //CBET_DEFINES_CUH
