#ifndef GPUEM_BESTERMATRIX_CUH
#define GPUEM_BESTERMATRIX_CUH

#include <cstdint>
#include <iostream>

#define _hd __host__ __device__
using uint = unsigned int;

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
template<typename T, uint32_t DIM, class Manager>
struct matrix_base : public Manager {
  T* data;
  uint32_t size;
  uint32_t dims[DIM];

  template<uint32_t... Args>
  matrix_base(Args... na) 
  : dims{na...}
  {
    static_assert(sizeof...(na) == DIM);
    
    size = dims[0];
    #pragma unroll DIM
    for(uint32_t i = 1; i < DIM; i++) {
      size *= dims[i];
    }

    if constexpr(std::is_same<Manager, cpu_managed>::value) {
      data = new T[size];
    } else {
      cudaChk(cudaMallocManaged(&data, num_bytes()))
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

  // _hd uint32_t size() const { return size; }
  _hd uint32_t num_bytes() const { return size * sizeof(T); }

  template<uint32_t... Args>
  _hd uint32_t get_index(Args... idx) const {
    static_assert(sizeof...(idx) == DIM);

    // Put arguments into an array. 
    // Is this necessary?
    const uint32_t indices[DIM] = {idx...};
    uint32_t global_index = 0;

    // Compute linear index from N-dimension input indices
    #pragma unroll DIM
    for(uint32_t i = 0; i < DIM ; i++) {
      auto val = indices[i];
      for(int j = i + 1; j < DIM; j++) {
        val += dims[j];
      }
      global_index += val;
    }
    return global_index;
  }

  _hd T& operator[] (uint32_t offset) { return data[offset]; }
  _hd const T& operator[] (uint32_t offset) const { return data[offset]; }

  template<uint32_t... Args> _hd T& operator() (Args... idx) { return data[get_index(idx...)]; }
  template<uint32_t... Args> _hd const T& operator() (Args... idx) const { return data[get_index(idx...)]; }
};

#endif //GPUEM_BESTERMATRIX_CUH