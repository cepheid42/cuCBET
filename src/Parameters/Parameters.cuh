#ifndef CBET_PARAMETERS_CUH
#define CBET_PARAMETERS_CUH

#include "../Utilities/Utilities.cuh"
#include "../Utilities/Vector2.cuh"

struct BeamParams {
  vec2<FPTYPE> b_norm;
  FPTYPE radius;
  FPTYPE sigma;
  FPTYPE intensity;
  FPTYPE omega;
  uint32_t nrays;
};

struct Parameters {
  vec2<FPTYPE> xy_min;
  vec2<FPTYPE> xy_max;
  FPTYPE CFL;
  FPTYPE dx;
  FPTYPE dy;
  FPTYPE dt;
  FPTYPE n_crit;
  uint32_t nx;
  uint32_t ny;
  uint32_t nt;
};

#endif //CBET_PARAMETERS_CUH
