#ifndef CBET_RAYS_CUH
#define CBET_RAYS_CUH

#include "../Utilities/Utilities.cuh"

struct Ray {
  vec3 ray_norm;
  float intensity;
};

#endif //CBET_RAYS_CUH
