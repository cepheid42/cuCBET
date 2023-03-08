#ifndef CUCBET_GRADIENTS_CUH
#define CUCBET_GRADIENTS_CUH

#include "../Utilities/Utilities.cuh"

__global__ void linear_electron_density_x(devMatrix<2>& eden, FPTYPE ncrit, FPTYPE eden_max, FPTYPE eden_min) {
  auto i = blockIdx.x;
  auto j = threadIdx.x;

  auto step = (eden_max - eden_min) / (eden.dims[0]);
  eden(i, j) = ncrit * (eden_min + i * step);
}

__global__ void gradient2D(devVector<2>& out,
                           devMatrix<2>& in,
                           FPTYPE dx, FPTYPE dy)
{
  auto i = blockIdx.x;
  auto j = threadIdx.x;

  FPTYPE x_deriv, y_deriv;
  // X derivative
  if (i == 0) {
    // left boundary, forward diff
    x_deriv = (-in(i + 2, j) + 4.0 * in(i + 1, j) - 3.0 * in(i, j)) / (2.0 * dx);
  } else if (i == in.dims[0] - 1) {
    // right boundary, backward diff
    x_deriv = (3.0 * in(i, j) - 4.0 * in(i - 1, j) + in(i - 2, j)) / (2.0 * dx);
  } else {
    // everywhere else, centered diff
    x_deriv = (in(i + 1, j) - in(i - 1, j)) / (2.0 * dx);
  }

  if (j == 0) {
    // bottom boundary, forward diff
    y_deriv = (-in(i, j + 2) + 4.0 * in(i, j + 1) - 3.0 * in(i, j)) / (2.0 * dy);
  } else if (j == in.dims[1] - 1) {
    // top boundary, backward diff
    y_deriv = (3.0 * in(i, j) - 4.0 * in(i, j - 1) + in(i, j - 2)) / (2.0 * dy);
  } else {
    // everywhere else, centered diff
    y_deriv = (in(i, j + 1) - in(i, j - 1)) / (2.0 * dy);
  }

  out(i, j) = vec2<FPTYPE>{x_deriv, y_deriv};
}

#endif //CUCBET_GRADIENTS_CUH