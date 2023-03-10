#ifndef BADCBET_TEST_BESTERMATRIX_CUH
#define BADCBET_TEST_BESTERMATRIX_CUH

#include <cassert>
#include "../Utilities/BesterMatrix.cuh"

using devVector   = matrix_base<float, 1, cuda_managed>;
using devArray    = matrix_base<float, 2, cuda_managed>;
using devMatrix3D = matrix_base<float, 3, cuda_managed>;
using devTensor   = matrix_base<float, 4, cuda_managed>;

template<typename Matrix>
__global__ void fill_gpu_matrix(Matrix& matrix, const float value) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  auto stride = blockDim.x * gridDim.x;

  // This also tests the operator[] for matrix class
  for (auto i = idx; i < matrix.size; i += stride) {
    matrix[i] = value;
  }
}

__global__ void add_gpu_vectors(devVector& c, const devVector& a, const devVector& b) {
  auto i = threadIdx.x;

  if (i == 0) {
    auto nDims = a.nDims;
    auto size = a.size;
    auto idx = a.get_index(i);
    auto data = a[idx];
    printf("add_vectors kernel tests: Matrix A nDims=%u size=%u, idx=%u, A[idx]=%f\n", nDims, size, idx, data);
  }

  if (i < c.dims[0]) {
    c(i) = a(i) + b(i);
  }
}

__global__ void add_gpu_arrays(devArray& c, const devArray& a, const devArray& b) {
  auto i = blockIdx.x;
  auto j = threadIdx.x;

  if (i == 0 && j == 0) {
    auto nDims = a.nDims;
    auto size = a.size;
    auto idx = a.get_index(i, j);
    auto data = a[idx];
    printf("add_vectors kernel tests: Matrix A nDims=%u size=%u, idx=%u, A[idx]=%f\n", nDims, size, idx, data);
  }

  if (i < c.dims[0] && j < c.dims[1]) {
    c(i, j) = a(i, j) + b(i, j);
  }
}

__global__ void add_gpu_matrices(devMatrix3D& c, const devMatrix3D& a, const devMatrix3D& b) {
  auto i = blockIdx.y;
  auto j = blockIdx.x;
  auto k = threadIdx.x;

  if (i == 0 && j == 0 && k == 0) {
    auto nDims = a.nDims;
    auto size = a.size;
    auto idx = a.get_index(i, j, k);
    auto data = a[idx];
    printf("add_matrices kernel tests: Matrix A nDims=%u size=%u, idx=%u, A[idx]=%f\n", nDims, size, idx, data);
  }

  if (i < c.dims[0] && j < c.dims[1] && k < c.dims[2]) {
    c(i, j, k) = a(i, j, k) + b(i, j, k);
  }
}

__global__ void add_gpu_tensor(devTensor& c, const devTensor& a, const devTensor& b) {
  auto i = blockIdx.y;
  auto j = blockIdx.x;
  auto k = threadIdx.x;

  if (i == 0 && j == 0 && k == 0) {
    auto nDims = a.nDims;
    auto size = a.size;
    auto idx = a.get_index(i, j, k, 0);
    auto data = a[idx];
    printf("add_matrices kernel tests: Matrix A nDims=%u size=%u, idx=%u, A[idx]=%f\n", nDims, size, idx, data);
  }

  if (i < c.dims[0] && j < c.dims[1] && k < c.dims[2]) {
    for (auto l = 0; l < c.dims[3]; l++) {
      c(i, j, k, l) = a(i, j, k, l) + b(i, j, k, l);
    }
  }
}

template<typename Matrix>
__global__ void assert_gpu_matrix(Matrix& matrix, const float value) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  auto stride = blockDim.x * gridDim.x;

  // This also tests the operator[] for matrix class
  for (auto i = idx; i < matrix.size; i += stride) {
    assert(matrix[i] == value);
  }
}

void test_vector(int nx) {
  auto Avec = new devVector(nx);
  auto Bvec = new devVector(nx);
  auto Cvec = new devVector(nx);

  fill_gpu_matrix<<<nx, nx>>>(*Avec, 1.0);
  fill_gpu_matrix<<<nx, nx>>>(*Bvec, 2.0);
  fill_gpu_matrix<<<nx, nx>>>(*Cvec, 0.0);

  assert_gpu_matrix<<<nx, nx>>>(*Avec, 1.0);
  assert_gpu_matrix<<<nx, nx>>>(*Bvec, 2.0);

  for(auto i = 0; i < nx; i++) {
    assert((*Cvec)(i) == 0.0);
  }

  add_gpu_vectors<<<1, nx>>>(*Cvec, *Avec, *Bvec);
  assert_gpu_matrix<<<nx, nx>>>(*Cvec, 3.0);
  cudaChk(cudaDeviceSynchronize())

  delete Avec;
  delete Bvec;
  delete Cvec;
}

void test_array(int nx, int ny) {
  auto Avec = new devArray(nx, ny);
  auto Bvec = new devArray(nx, ny);
  auto Cvec = new devArray(nx, ny);

  fill_gpu_matrix<<<nx, nx>>>(*Avec, 1.0);
  fill_gpu_matrix<<<nx, nx>>>(*Bvec, 2.0);
  fill_gpu_matrix<<<nx, nx>>>(*Cvec, 0.0);

  assert_gpu_matrix<<<nx, nx>>>(*Avec, 1.0);
  assert_gpu_matrix<<<nx, nx>>>(*Bvec, 2.0);

  for(auto i = 0; i < nx; i++) {
    for(auto j = 0; j < ny; j++) {
      assert((*Cvec)(i, j) == 0.0);
    }
  }

  add_gpu_arrays<<<nx, nx>>>(*Cvec, *Avec, *Bvec);
  assert_gpu_matrix<<<nx, nx>>>(*Cvec, 3.0);
  cudaChk(cudaDeviceSynchronize())

  delete Avec;
  delete Bvec;
  delete Cvec;
}

void test_matrix(int nx, int ny, int nz) {
  auto Avec = new devMatrix3D(nx, ny, nz);
  auto Bvec = new devMatrix3D(nx, ny, nz);
  auto Cvec = new devMatrix3D(nx, ny, nz);

  fill_gpu_matrix<<<nx, nx>>>(*Avec, 1.0);
  fill_gpu_matrix<<<nx, nx>>>(*Bvec, 2.0);
  fill_gpu_matrix<<<nx, nx>>>(*Cvec, 0.0);

  assert_gpu_matrix<<<nx, nx>>>(*Avec, 1.0);
  assert_gpu_matrix<<<nx, nx>>>(*Bvec, 2.0);

  for(auto i = 0; i < nx; i++) {
    for(auto j = 0; j < ny; j++) {
      for(auto k = 0; k < nz; k++) {
        assert((*Cvec)(i, j, k) == 0.0);
      }
    }
  }

  dim3 blocks{uint32_t(nx), uint32_t(ny)};
  add_gpu_matrices<<<blocks, nx>>>(*Cvec, *Avec, *Bvec);
  assert_gpu_matrix<<<nx, nx>>>(*Cvec, 3.0);
  cudaChk(cudaDeviceSynchronize())

  delete Avec;
  delete Bvec;
  delete Cvec;
}

void test_tensor(int nx, int ny, int nz, int nw) {
  auto Avec = new devTensor(nx, ny, nz, nw);
  auto Bvec = new devTensor(nx, ny, nz, nw);
  auto Cvec = new devTensor(nx, ny, nz, nw);

  fill_gpu_matrix<<<nx, 4 * nx>>>(*Avec, 1.0);
  fill_gpu_matrix<<<nx, 4 * nx>>>(*Bvec, 2.0);
  fill_gpu_matrix<<<nx, 4 * nx>>>(*Cvec, 0.0);

  assert_gpu_matrix<<<nx, 4 * nx>>>(*Avec, 1.0);
  assert_gpu_matrix<<<nx, 4 * nx>>>(*Bvec, 2.0);

  for(auto i = 0; i < nx; i++) {
    for(auto j = 0; j < ny; j++) {
      for(auto k = 0; k < nz; k++) {
        for(auto l = 0; l < nw; l++) {
          assert((*Cvec)(i, j, k, l) == 0.0);
        }
      }
    }
  }

  dim3 blocks{uint32_t(nx), uint32_t(ny)};
  add_gpu_tensor<<<blocks, nx>>>(*Cvec, *Avec, *Bvec);
  assert_gpu_matrix<<<nx, 4 * nx>>>(*Cvec, 3.0);
  cudaChk(cudaDeviceSynchronize())

  delete Avec;
  delete Bvec;
  delete Cvec;
}

void test_bester_matrix() {
  auto nx = 64;
  auto ny = nx;
  auto nz = nx;
  auto nw = nx;

  test_vector(nx);
  test_array(nx, ny);
  test_matrix(nx, ny, nz);
  test_tensor(nx, ny, nz, nw);
}

#endif //BADCBET_TEST_BESTERMATRIX_CUH
