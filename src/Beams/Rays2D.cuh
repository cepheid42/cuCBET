#ifndef CUCBET_RAYS2D_CUH
#define CUCBET_RAYS2D_CUH

#include "../Utilities/Utilities.cuh"

struct Ray2D {
  float controls[6];
  float intensity;

  Ray2D() = default;

  _hd Ray(const Vector2<float>& A, const Vector2<float>& B, float _intensity)
  : controls{xa, xb, xc, ya, yb, yc},
    intensity(_intensity)
  {}

  _hd void update_B(float xb, float yb) {
    controls[1] = xb;
    controls[4] = yb;
  }

  _hd void update_C(float xc, float yc) {
    controls[2] = xc;
    controls[5] = yc;
  }

  _hd Vector2<float> eval(float t) const {
    auto omt = 1.0f - t;
    auto px = (omt * omt * controls[0]) + (2.0 * t * omt * controls[1]) + (t * t * controls[2]);
    auto py = (omt * omt * controls[3]) + (2.0 * t * omt * controls[4]) + (t * t * controls[5]);
    return {px, py};
  }
};

#endif //CUCBET_RAYS2D_CUH
