#ifndef CBET_BEZIER_CUH
#define CBET_BEZIER_CUH

#include "../Utilities/Vector3.cuh"

__constant__ float PmatT[9] = {1.0, 0.0, 0.0, -2.0, 2.0, 0.0, 1.0, -2.0, 1.0};
__constant__ float BmatT[9] = {1.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 1.0};

//--------------------------------------------------
// Quadratic Bezier Curve Class
struct BezierCurve {
  float controls[9];

  _hd BezierCurve(Vector3<float> A, Vector3<float> B, Vector3<float> C) 
  : controls{A[0], B[0], C[0], A[1], B[1], C[1], A[2], B[2], C[2]}
  {}

  // Update end control point
  _hd void update_C_control(Vector3<float> newC) {
      controls[2] = newC[0];
      controls[5] = newC[1];
      controls[8] = newC[2];
  }
};

//--------------------------------------------------
// Bezier Curve Evaluation Function
_hd Vector3<float> eval(const BezierCurve curve, float t) {
  const float tpow[3] = {1.0, t, t * t};
  Vector3<float> p();

  for (auto row = 0; row < 3; row++) {
    for (auto col = 0; col < 3; col++) {
      p[0] += curve.controls[row] * PmatT[row]
    }
  }
}


#endif //CBET_BEZIER_CUH