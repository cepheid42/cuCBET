#ifndef CBET_PARAMETERS_CUH
#define CBET_PARAMETERS_CUH

#include "../Utilities/Utilities.cuh"
#include "../Utilities/Vector2.cuh"

struct BeamParams {
  vec2<FPTYPE> b_norm;
  FPTYPE radius;
  FPTYPE sigma;
  FPTYPE intensity;
  FPTYPE lambda;
  uint32_t nrays;
};

struct Parameters {
  uint32_t nx;
  uint32_t ny;
  uint32_t nt;
  FPTYPE CFL;
  FPTYPE dx;
  FPTYPE dy;
  vec2<FPTYPE> x;
  vec2<FPTYPE> y;
};

#endif //CBET_PARAMETERS_CUH
