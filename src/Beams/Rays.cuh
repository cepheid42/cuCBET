#ifndef CBET_RAYS_CUH
#define CBET_RAYS_CUH

#include "../Utilities/Utilities.cuh"

struct Ray {
  float controls[9];
  float intensity;

  Ray() = default;
  
  _hd Ray(const Vector3<float>& A, const Vector3<float>& B, const Vector3<float>& C, float _intensity) 
  : controls{A[0], B[0], C[0], 
             A[1], B[1], C[1], 
             A[2], B[2], C[2]},
    intensity(_intensity)
  {}

  // Update end control point
  _hd void update_C_control(Vector3<float> newC) {
    controls[2] = newC[0];
    controls[5] = newC[1];
    controls[8] = newC[2];
  }

  // Evaluate P(t) = (1 - t)^2 * A + 2t(1-t) * B + t^2 * C
  _hd Vector3<float> eval(float t) {
    float omt = 1.0 - t;
    float px = (omt * omt * controls[0]) + (2.0 * t * omt * controls[1]) + (t * t * controls[2]);
    float py = (omt * omt * controls[3]) + (2.0 * t * omt * controls[4]) + (t * t * controls[5]);
    float pz = (omt * omt * controls[6]) + (2.0 * t * omt * controls[7]) + (t * t * controls[8]);

    return {px, py, pz};
  }
};

#endif //CBET_RAYS_CUH
