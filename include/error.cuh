//
// Created by cepheid on 3/7/24.
//

#ifndef CUCBET_ERROR_CUH
#define CUCBET_ERROR_CUH

#include <iostream>

#define cudaChk(ans)  { gpuAssert((ans), __FILE__, __LINE__, __PRETTY_FUNCTION__); } static_assert(true, "")

inline void gpuAssert(cudaError_t code, const char* file, int line, const char* func,  bool abort=true) {
  if (code != cudaSuccess) {
    std::cout << "["
              << file
              << ":"
              << line
              << "]("
              << func
              << "): "
              << cudaGetErrorString(code)
              << std::endl;
    if (abort) { exit(code); }
  }
}

#endif //CUCBET_ERROR_CUH
