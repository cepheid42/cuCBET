#ifndef CBET_RAYS_CUH
#define CBET_RAYS_CUH

#include "./Bezier.cuh"
#include "../Utilities/Utilities.cuh"

struct Ray : public BezierCurve {
  float A;
};

#endif //CBET_RAYS_CUH
