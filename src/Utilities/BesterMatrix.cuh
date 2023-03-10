#ifndef GPUEM_BESTERMATRIX_CUH
#define GPUEM_BESTERMATRIX_CUH

#include <cstdint>
#include <iostream>

#define hd __host__ __device__

#define cudaChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    std::cout << "[" << file << ":" << line << "] GPU Error: " << cudaGetErrorString(code) << std::endl;
    if (abort) { exit(code); }
  }
}

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

/****************************************/
/********** Matrix Base Class ***********/
template<typename T, int DIM, class Manager>
struct matrix_base : public Manager {
  T* data = nullptr;
  uint32_t size{};
  uint32_t nDims{};
  uint32_t dims[DIM]{};

  template<typename... Args>
  constexpr explicit matrix_base(const Args... na) : nDims{DIM}, dims{uint32_t(na)...} {
    static_assert(sizeof...(na) == DIM);
    size = dims[0];

    for(uint32_t i = 1; i < DIM; i++) { size *= dims[i]; }

    if constexpr(std::is_same<Manager, cpu_managed>::value) {
      data = new T[size];
    } else {
      cudaChk(cudaMallocManaged(&data, size * sizeof(T)))
      cudaChk(cudaDeviceSynchronize())
    }
  }

  ~matrix_base() {
    if constexpr(std::is_same<Manager, cpu_managed>::value) {
      delete[] data;
    } else {
      cudaChk(cudaDeviceSynchronize())
      cudaChk(cudaFree(data))
    }
  }

  template<typename... Args>
  hd uint32_t get_index(const Args... idx) const {
    // Compute linear index from N-dimension input indices
    static_assert(sizeof...(idx) == DIM);
    uint32_t indices[DIM] = {uint32_t(idx)...};
    uint32_t global_index = indices[0];
    for (int i = 1; i < DIM; ++i) {
      global_index = indices[i] + dims[i] * global_index;
    }
    return global_index;
  }

  hd       T& operator[] (uint32_t offset)       { return data[offset]; }
  hd const T& operator[] (uint32_t offset) const { return data[offset]; }
  template<typename... Args> hd       T& operator() (const Args... idx)       { return data[get_index(idx...)]; }
  template<typename... Args> hd const T& operator() (const Args... idx) const { return data[get_index(idx...)]; }
};

#endif //GPUEM_BESTERMATRIX_CUH