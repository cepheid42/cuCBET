#ifndef CBET_RAYS_CUH
#define CBET_RAYS_CUH

#include "../Utilities/Utilities.cuh"

template<typename T>
struct Ray3D {
  T controls[4][3];
  T weights[4];
  T intensity;

  Ray3D() = default;

  hd Ray3D(const vec3<T>& P0, const vec3<T>& P1, const vec3<T>& P2, const vec3<T>& P3, float _intensity)
  : controls{{P0[0], P0[1], P0[2]}, {P1[0], P1[1], P1[2]}, {P2[0], P2[1], P2[2]}, {P3[0], P3[1], P3[2]}},
    intensity(_intensity)
  {}

  hd vec3<T> eval(T t) {
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

    vec3<T> result{0.0, 0.0, 0.0};
    for (int i = 0; i < 4; i++) {
      result[0] += weighted_t[i] * controls[i][0];
      result[1] += weighted_t[i] * controls[i][1];
      result[2] += weighted_t[i] * controls[i][2];
    }

    result /= denominator;
    return result;
  }
};


#endif //CBET_RAYS_CUH
