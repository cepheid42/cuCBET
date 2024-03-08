#ifndef GPUEM_BESTERMATRIX_CUH
#define GPUEM_BESTERMATRIX_CUH

#include <cstdint>
#include <iostream>
#include <cassert>

#include "error.cuh"
#include "smart_pointer.cuh"

/*************************************/
/********** 2D Array Class ***********/
template<typename T>
struct Array2d {
  dev_ptr<T> data;
  size_t dims[2];

  explicit Array2d(size_t nx, size_t ny)
  : data{make_dev_ptr<T[]>(nx * ny)},
    dims{nx, ny}
  {}

  ~Array2d() = default;

  [[nodiscard]] inline size_t get_index(size_t i, size_t j) const {
    assert(i < dims[0] && j < dims[1]);
    return j + (dims[1] * i);
  }

        T& operator[] (size_t offset)       { return data[offset]; }
  const T& operator[] (size_t offset) const { return data[offset]; }
        T& operator() (size_t i, size_t j)       { return data[get_index(i, j)]; }
  const T& operator() (size_t i, size_t j) const { return data[get_index(i, j)]; }
};

#endif //GPUEM_BESTERMATRIX_CUH