#ifndef CBET_NEURALNET_CUH
#define CBET_NEURALNET_CUH

#include "./Utilities/BestMatrix.cuh"

using devMatrix = matrix_base<float, cuda_managed>;

__device__ void matmul(devMatrix& A, devMatrix& B);

__device__ void diff(devMatrix& A, devMatrix& B);

__device__ void mdot(devMatrix& A, devMatrix& B);



_global__ void update_NN(devMatrix& V, devMatrix& X, devMatrix& T, devMatrix& W, ActivationFunc h) {
  auto i = threadIdx.z + blockDim.z * blockIdx.z;
  auto j = threadIdx.y + blockDim.y * blockIdx.y;
  auto k = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ float shmem[];
  auto* x = ...;
  auto* v = ...;
  __syncthreads();

  z = h(matmul(X_prepend_one, V));
  y = matmul(Z_prepend_ones, W);
  dw = diff(T, Y);
  dv = mdot(matmul(dw, W_hat_transpose), (1 - SQR(Z));
  W = W + (ro / (N * K)) * matmul(Z_transpose_prepend_ones, dw);
  V = V + (rh / (N * K)) * matmul(X_transpose_prepend_ones, dv);
}


#endif //CBET_NEURALNET_CUH
