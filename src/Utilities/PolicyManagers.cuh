#ifndef CUCBET_POLICYMANAGERS_CUH
#define CUCBET_POLICYMANAGERS_CUH

#include <cstdint>
#include "Defines.cuh"

/****************************************/
/****** Management Policy Classes *******/
struct cuda_managed {
  void* operator new(size_t len) {
    void* ptr;
    cudaChk(cudaMallocManaged(&ptr, len))
    cudaChk(cudaDeviceSynchronize())
    return ptr;
  }

  void operator delete(void* ptr) {
    cudaChk(cudaDeviceSynchronize())
    cudaChk(cudaFree(ptr))
  }
};

struct cpu_managed {
  void* operator new(size_t len) {
    // avoid std::malloc(0) which may return nullptr on success
    if (len == 0) { ++len; }
    void* ptr = std::malloc(len);
    if (ptr) { return ptr; }
    throw std::bad_alloc{};
  }

  void operator delete(void* ptr) { std::free(ptr); }
};

#endif //CUCBET_POLICYMANAGERS_CUH
