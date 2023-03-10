#ifndef CUCBET_GRADIENTS_CUH
#define CUCBET_GRADIENTS_CUH

#include "../Utilities/Utilities.cuh"

void linear_electron_density_x(devMatrix<2>& eden, FPTYPE eden_max, FPTYPE eden_min) {
  auto step = (eden_max - eden_min) / (eden.dims[0] - 1);
  for (uint32_t i = 0; i < eden.dims[0]; i++) {
    for (uint32_t j = 0; j < eden.dims[1]; j++) {
      eden(i, j) = eden_min + i * step;
    }
  }
}

void gradient2D(devVector<2>& out, const devMatrix<2>& in, FPTYPE dx, FPTYPE dy) {
  auto EPSILON = std::numeric_limits<FPTYPE>::epsilon();

  /***** X Derivative *****/
  auto xderiv = new FPTYPE[in.size];
  // Do first row
  auto first_x = 0;
  for (uint32_t j = 0; j < in.dims[1]; ++j) {
    auto first  = -3.0 * in(0, j);
    auto second =  4.0 * in(1, j);
    auto third  = -1.0 * in(2, j);
    auto last = first + second + third;
    if (abs(last) <= EPSILON) {
      last = 0.0;
    }
    xderiv[j + (in.dims[1] * first_x)] = last / (2.0 * dx);
  }

  // Do last row
  auto last_x = in.dims[0] - 1;
  for (uint32_t j = 0; j < in.dims[1]; ++j) {
    auto first  =  3.0 * in(last_x, j);
    auto second = -4.0 * in(last_x - 1, j);
    auto third  =  1.0 * in(last_x - 2, j);
    auto last = first + second + third;
    if (abs(last) <= EPSILON) {
      last = 0.0;
    }
    xderiv[j + (in.dims[1] * last_x)] = last / (2.0 * dx);
  }

  // Do all remaining interior rows
  for (uint32_t i = 1; i < in.dims[0] - 1; ++i) {
    for (uint32_t j = 0; j < in.dims[1]; ++j) {
      xderiv[j + (in.dims[1] * i)] = (in(i + 1, j) - in(i - 1, j)) / (2.0 * dx);
    }
  }
  
  /***** Y Derivative *****/
  auto yderiv = new FPTYPE[in.size];
  // Do first column
  auto first_y = 0;
  for (uint32_t i = 0; i < in.dims[0]; ++i) {
    auto first  = -3.0 * in(i, 0);
    auto second =  4.0 * in(i, 1);
    auto third  = -1.0 * in(i, 2);
    auto last = first + second + third;
    if (abs(last) <= EPSILON) {
      last = 0.0;
    }
    yderiv[first_y + (in.dims[1] * i)] = last / (2.0 * dy);
  }

  // Do last column
  auto last_y = in.dims[1] - 1;
  for (uint32_t i = 0; i < in.dims[0]; ++i) {
    auto first  =  3.0 * in(i, last_y);
    auto second = -4.0 * in(i, last_y - 1);
    auto third  =  1.0 * in(i, last_y - 2);
    auto last = first + second + third;
    if (abs(last) <= EPSILON) {
      last = 0.0;
    }
    yderiv[last_y + (in.dims[1] * i)] = last / (2.0 * dy);
  }

  // Do all remaining interior columns
  for (uint32_t i = 0; i < in.dims[0]; ++i) {
    for (uint32_t j = 1; j < in.dims[1] - 1; ++j) {
      yderiv[j + (in.dims[1] * i)] = (in(i, j + 1) - in(i, j - 1)) / (2.0 * dy);
    }
  }

  /***** Fill Output *****/
  for (uint32_t i = 0; i < in.dims[0]; ++i) {
    for (uint32_t j = 0; j < in.dims[1]; ++j) {
      auto x_val = xderiv[j + (in.dims[1] * i)];
      auto y_val = yderiv[j + (in.dims[1] * i)];
      out(i, j) = {x_val, y_val};
    }
  }
}


#endif //CUCBET_GRADIENTS_CUH