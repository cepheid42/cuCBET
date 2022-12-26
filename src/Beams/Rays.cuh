#ifndef CBET_RAYS_CUH
#define CBET_RAYS_CUH

#include "./Bezier.cuh"
#include "../Utilities/Utilities.cuh"

template<typename T>
struct Ray : public BezierCurve {
  float intensity;

  Ray(Vector3<T> origin, Vector3<T> center, Vector3<T> end, T intensity) 
  : BezierCurve{origin, center, end}, intensity(intensity)
  {}
};

#endif //CBET_RAYS_CUH
