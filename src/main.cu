#include "Parameters/Parameters.cuh"
#include "Beams/Beams.cuh"

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
//#include "Interpolation/Interpolator.c/

int main() {
  constexpr auto Nx = 10;
  constexpr auto nrays = 5.0f;
  constexpr auto beam_min = -3.0E-4f;
  constexpr auto beam_max = 3.0E-4f;
  constexpr auto sigma = 1.7E-4f;

  constexpr auto dx = beam_max - beam_min;




  return 0;
}
