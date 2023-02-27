#ifndef CUCBET_RAY2D_CUH
#define CUCBET_RAY2D_CUH

#include "../Utilities/Vector2.cuh"

#define hd __host__ __device__

template<typename T>
struct Ray2D {
  T controls[4][2];
  T weights[4];
  T intensity;

  Ray2D() = default;

  hd Ray2D(const vec2<T>& P0, const vec2<T>& P1, const vec2<T>& P2, const vec2<T>& P3, float _intensity)
  : controls{{P0[0], P0[1]}, 
             {P1[0], P1[1]}, 
             {P2[0], P2[1]}, 
             {P3[0], P3[1]}},
    intensity(_intensity)
  {}

  hd vec2<T> eval(T t) {
    const T ta[4] = {1, SQR(t), CUBE(t), SQR(T) * SQR(t)};
    const T w[4][4] = {{weights[0], -3.0 * weights[0],  3.0 * weights[0],       -weights[0]},
                       {       0.0,  3.0 * weights[1], -6.0 * weights[1],  3.0 * weights[1]},
                       {       0.0,               0.0,  3.0 * weights[2], -3.0 * weights[2]},
                       {       0.0,               0.0,               0.0,        weights[3]}};

    T weighted_t[4]{0.0};
    T denominator = 0.0;

    for (int i = 0; i < 4; ++i) {
      auto temp = 0.0;
      for (int j = 0; j < 4; ++j) {
        temp += ta[j] * w[i][j];
      }
      weighted_t[i] = temp;
      denominator += temp;
    }

    vec2<T> result{0.0, 0.0};
    for (int i = 0; i < 4; i++) {
      result[0] += weighted_t[i] * controls[i][0];
      result[1] += weighted_t[i] * controls[i][1];
    }
    result /= denominator;
    return result;
  }
};

#endif //CUCBET_RAY2D_CUH
