#ifndef GPUEM_BESTMATRIX_CUH
#define GPUEM_BESTMATRIX_CUH

//#include <cstdint>
//#include "Defines.cuh"
#include "PolicyManagers.cuh"

/****************************************/
/********** Matrix Base Class ***********/
template<typename T, class Manager>
struct matrix_base : public Manager {
  T* data;
  uint32_t x, y, z;

  explicit matrix_base(dim3 dims)
  : x(dims.x), y(dims.y), z(dims.z)
  {
    if constexpr(std::is_same<Manager, cpu_managed>::value) {
      data = new T[size()];
    } else {
      cudaChk(cudaMallocManaged(&data, num_bytes()))
      cudaChk(cudaDeviceSynchronize())
    }
  }

  explicit matrix_base(uint32_t nx, uint32_t ny = 1, uint32_t nz = 1)
  : matrix_base(dim3(nx, ny, nz)) {}

  ~matrix_base() {
    if constexpr(std::is_same<Manager, cpu_managed>::value) {
      delete[] data;
    } else {
      cudaChk(cudaDeviceSynchronize())
      cudaChk(cudaFree(data))
    }
  }

  _hd uint32_t size() const { return x * y * z; }
  _hd uint32_t num_bytes() const { return size() * sizeof(T); }

  _hd uint32_t get_index(uint32_t i, uint32_t j) const { return (y * i) + j; }
  _hd uint32_t get_index(uint32_t i, uint32_t j, uint32_t k) const { return (y * z * i) + (z * j) + k; }

  // 1D indexing
  _hd T& operator[] (uint32_t offset) { return data[offset]; }
  _hd const T& operator[] (uint32_t offset) const { return data[offset]; }

  // 2D indexing
  _hd T& operator() (uint32_t i, uint32_t j) { return data[get_index(i, j)]; }
  _hd const T& operator() (uint32_t i, uint32_t j) const { return data[get_index(i, j)]; }

  // 3D indexing
  _hd T& operator() (uint32_t i, uint32_t j, uint32_t k) { return data[get_index(i, j, k)]; }
  _hd const T& operator() (uint32_t i, uint32_t j, uint32_t k) const { return data[get_index(i, j, k)]; }
};

#endif //GPUEM_BESTMATRIX_CUH