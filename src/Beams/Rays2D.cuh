#ifndef CUCBET_RAYS2D_CUH
#define CUCBET_RAYS2D_CUH

#include "../Utilities/Utilities.cuh"

using vec2f = vec2<float>;

struct Ray2D {
  float controls[8];
  float intensity;

  Ray2D() = default;

  _hd Ray2D(const vec2f& A0, const vec2f& A1, const vec2f& A2, const vec2f& A3, float _intensity)
  : controls{A0[0], A0[1], A1[0], A1[1], A2[0], A2[1], A3[0], A3[1]},
    intensity(_intensity)
  {}

  _hd void update_B(float xb, float yb) {
    // controls[1] = xb;
    // controls[4] = yb;
  }

  _hd void update_C(float xc, float yc) {
    // controls[2] = xc;
    // controls[5] = yc;
  }

  _hd Vector2<float> eval(float t) const {

  }
};

#endif //CUCBET_RAYS2D_CUH
