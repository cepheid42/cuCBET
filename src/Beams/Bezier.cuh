#ifndef CBET_BEZIER_CUH
#define CBET_BEZIER_CUH

#include "../Utilities/Vector3.cuh"

//--------------------------------------------------
// Quadratic Bezier Curve Class
struct BezierCurve {
  float controls[9];

  _hd BezierCurve(Vector3<float> A, Vector3<float> B, Vector3<float> C) 
  : controls{A[0], B[0], C[0], 
             A[1], B[1], C[1], 
             A[2], B[2], C[2]}
  {}

  // Update end control point
  _hd void update_C_control(Vector3<float> newC) {
    controls[2] = newC[0];
    controls[5] = newC[1];
    controls[8] = newC[2];
  }

  // Evaluate P(t) = (1 - t)^2 * A + 2t(1-t) * B + t^2 * C
  _hd Vector3<float> eval(const BezierCurve curve, float t) {
    auto omt = 1.0 - t;
    auto px = (SQR(omt) * controls[0]) + (2.0 * t * omt * controls[1]) + (SQR(t) * controls[2]);
    auto py = (SQR(omt) * controls[3]) + (2.0 * t * omt * controls[4]) + (SQR(t) * controls[5]);
    auto pz = (SQR(omt) * controls[6]) + (2.0 * t * omt * controls[7]) + (SQR(t) * controls[8]);

    return {px, py, pz};
  }
};

#endif //CBET_BEZIER_CUH