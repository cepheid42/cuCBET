#ifndef CBET_PARAMETERS_CUH
#define CBET_PARAMETERS_CUH

#include "../Utilities/Utilities.cuh"

struct BeamParams {
  FPTYPE dist;
  FPTYPE radius;
  FPTYPE sigma;
  FPTYPE intensity;
  uint32_t nrays;
};

struct Parameters {
  BeamParams beam;
  uint32_t nx;
  uint32_t ny;
  uint32_t nt;
};

#endif //CBET_PARAMETERS_CUH
