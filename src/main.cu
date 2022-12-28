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

#include <type_traits>

#include "./Beams/Beams.cuh"

int main() {

  std::cout << std::is_pod<Vector3<float>>::value << std::endl;
  std::cout << std::is_pod<BezierCurve>::value << std::endl;
  std::cout << std::is_pod<Ray>::value << std::endl;

  vec3 bnorm{-1.0, 0.0, 0.0};
  Beam b1 = new Beam(0, bnorm, 1.0, 0.1, 10.0, 1.0);

  beam_to_csv


  return 0;
}
