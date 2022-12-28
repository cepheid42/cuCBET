//#include "Parameters/Parameters.cuh"

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

#include "./Utilities/Utilities.cuh"
#include "./Beams/Beams.cuh"

int main() {
  vec3 bnorm1{-1.0, 0.0, 0.0};
  Beam* b1 = new Beam(0, bnorm1, 1.0, 0.1, 10.0, 1.0);

  vec3 bnorm2{0.0, -1.0, 0.0};
  Beam* b2 = new Beam(1, bnorm2, 1.0, 0.1, 10.0, 1.0);

  vec3 bnorm3{-0.347252784, -0.782219564,	0.517250479};
  Beam* b3 = new Beam(2, bnorm3, 1.0, 0.1, 10.0, 1.0);

  beam_to_csv(*b1, "./beam1.csv");
  beam_to_csv(*b2, "./beam2.csv");
  beam_to_csv(*b3, "./beam3.csv");

  return 0;
}
