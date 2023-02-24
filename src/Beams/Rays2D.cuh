#ifndef CUCBET_RPYS2D_CUH
#define CUCBET_RPYS2D_CUH

#include "../Utilities/Vector3.cuh"

#define hd __host__ __device__

template<typename T>
struct Ray2D {
  float controls[12];
  float intensity;

  Ray2D() = default;

  hd Ray2D(const vec3<T>& P0, const vec3<T>& P1, const vec3<T>& P2, const vec3<T>& P3, float _intensity)
  : controls{P0[0], P0[1], P0[2], P1[0], P1[1], P1[2], P2[0], P2[1], P2[2], P3[0], P3[1], P3[2]},
    intensity(_intensity)
  {}

  hd vec3<T> eval
};

template<typename T>


#endif //CUCBET_RPYS2D_CUH
