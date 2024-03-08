#ifndef CUCBET_RAY2D_CUH
#define CUCBET_RAY2D_CUH

#include "vector2.cuh"

#define hd __host__ __device__

template<typename T>
struct Ray2D {
  vec2<T> controls[4];
  T intensity;

  Ray2D() = default;

  hd Ray2D(const vec2<T>& P0, const vec2<T>& P1, const vec2<T>& P2, const vec2<T>& P3, T _intensity)
  : controls{P0, P1, P2, P3},
    intensity(_intensity)
  {}

  hd vec2<T> eval(T t) {
    auto t2 = SQR(t);
    auto t3 = CUBE(t);
    auto omt = 1.0 - t;
    auto omt2 = SQR(omt);
    auto omt3 = CUBE(omt);
    return (omt3 * controls[0]) + (3.0 * t * omt2 * controls[1]) + (3.0 * t2 * omt * controls[2]) + (t3 * controls[3]);
  }

  hd vec2<T> first_deriv(T t) {
    auto omt = 1.0 - t;
    auto P0 = controls[1] - controls[0];
    auto P1 = controls[2] - controls[1];
    auto P2 = controls[3] - controls[2];
    return (3.0 * SQR(omt) * P0) + (6.0 * t * omt * P1) + (3.0 * SQR(t) * P2);
  }
};

#endif //CUCBET_RAY2D_CUH
