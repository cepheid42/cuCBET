#ifndef CBET_RAYS_CUH
#define CBET_RAYS_CUH

#include "./Bezier.cuh"
#include "../Utilities/Utilities.cuh"

struct Ray : public BezierCurve {
  float intensity;

  Ray(Vector3<float> origin, Vector3<float> center, Vector3<float> end, float intensity) 
  : BezierCurve(origin, center, end), intensity(intensity)
  {}
};

#endif //CBET_RAYS_CUH
