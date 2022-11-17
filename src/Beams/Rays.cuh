#ifndef CBET_RAYS_CUH
#define CBET_RAYS_CUH

#include "../Utilities/Utilities.cuh"

struct Ray {
  dim3 ray_origin;
  vec3 k_vec;
  float amp;
};

#endif //CBET_RAYS_CUH
