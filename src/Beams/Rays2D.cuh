#ifndef CUCBET_RPYS2D_CUH
#define CUCBET_RPYS2D_CUH

#include "../Utilities/Vector2.cuh"

#define hd __host__ __device__

template<typename T>
struct Ray2D {
  float controls[8];
  float intensity;

  Ray2D() = default;

  hd Ray2D(const vec2<T>& P0, const vec2<T>& P1, const vec2<T>& P2, const vec2<T>& P3, float _intensity)
  : controls{P0[0], P0[1], P1[0], P1[1], P2[0], P2[1], P3[0], P3[1]},
    intensity(_intensity)
  {}
  
  hd void implicitize() {
    auto a0 = controls[]
    auto a1 =;
    auto a2 =;
    auto b0 =;
    auto b1 = ;
    auto b2 = ;
  }
};

#endif //CUCBET_RPYS2D_CUH
