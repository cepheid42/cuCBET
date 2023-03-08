#ifndef CBET_INTERP2D_CUH
#define CBET_INTERP2D_CUH

#include "../Utilities/Utilities.cuh"

hd vec2<FPTYPE> interp2D(const devVector<2>& data,
                         const vec2<FPTYPE>& point,
                         const vec2<FPTYPE> mins,
                         FPTYPE dx, FPTYPE dy)
{
  auto m = (point[0] - mins[0]) / dx;
  auto n = (point[1] - mins[1]) / dy;

  auto i = floor(m);
  auto j = floor(n);

  auto alpha = m - i;
  auto beta = n - j;

  i = static_cast<uint32_t>(i);
  j = static_cast<uint32_t>(i);

  auto p1 = (1.0 - alpha) * (1.0 - beta) * data(i, j);
  auto p2 = alpha * (1.0 - beta) * data(i + 1, j);
  auto p3 = (1.0 - alpha) * beta * data(i, j + 1);
  auto p4 = alpha * beta * data(i + 1, j + 1);

  return (p1 + p2 + p3 + p4);
}

#endif //CBET_INTERP2D_CUH
