//#include "Parameters/Parameters.cuh"
//#include "Beams/Beams.cuh"

//uint32_t* marked  = 200 * nx * ny * nz * sizeof(uint32_t);
//uint32_t* counter = nx * ny * nz * sizeof(uint32_t);
//float* eden       = nx * ny * nz * sizeof(float);
//float* etemp      = nx * ny * nz * sizeof(float);
//float* machnum    = 3 * nx * ny * nz * sizeof(float);
//
//float* intensity_init = nbeams * nrays * sizeof(float);
//float* kvec           = nbeams * nrays * ncrossings * sizeof(float);
//float* i_b            = nbeams * nrays * ncrossings * sizeof(float);
//float* ray_areas      = nbeams * nrays * ncrossings * sizeof(float);
//float* polar_angle    = nbeams * nrays * ncrossings * sizeof(float);
//float* wMult          = 2 * nbeams * nrays * ncrossings * sizeof(float);

//#include "./Utilities/Utilities.cuh"
#include "./Interpolation/Interpolator.cuh"

int main() {
  constexpr auto Nx = 101;
  auto dx = 6.28f / (Nx - 1);
  auto np = 51;
  auto dp = 6.28f / float(np - 1);

  auto* xs = new float[np];
  auto* ys = new float[np];

  for (auto i = 0; i < np; i++) {
    xs[i] = float(i) * dp;
    ys[i] = std::sin(xs[i]);
  }

//  for (auto i = 0; i < np; i++) {
//    std::cout << xs[i] << ", " << ys[i] << std::endl;
//  }

  Interpolator<LINEAR, cpu_managed> linterp(xs, ys, np);

  for(auto i = 0; i < Nx; i++) {
    auto x = linterp(float(i) * dx);
//    auto x = float(i) * dx;
//    std::cout << x << ", " << linterp(x) << std::endl;
  }

  return 0;
}
