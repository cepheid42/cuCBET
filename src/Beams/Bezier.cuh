#ifndef CBET_BEZIER_CUH
#define CBET_BEZIER_CUH

#include "../Utilities/Vector3.cuh"

__constant__ Vector3<float> PmatT[3] = {{1.0, 0.0, 0.0}, {-2.0, 2.0, 0.0}, {1.0, -2.0, 1.0}};
__constant__ Vector3<float> BmatT[3] = {{1.0, 1.0, 0.0}, {0.0, -2.0, 0.0}, {0.0, 1.0, 1.0}};

//--------------------------------------------------
// Quadratic Bezier Curve Class
template<typename T>
struct BezierCurve {
  Vector3<T> controls[3];

  _hd BezierCurve(Vector3<T> A, Vector3<T> B, Vector3<T> C) 
  : controls{Vector3<T>{A[0], B[0], C[0]}, Vector3<T>{A[1], B[1], C[1]}, Vector3<T>{A[2], B[2], C[2]}}
  {};

  // Update end control point
  _hd void update_C_control(Vector3<T> newC) {
      controls[0][2] = newC[0];
      controls[1][2] = newC[1];
      controls[2][2] = newC[2];
  };

  // _hd void update_B_control(T t = 0.5) {
  //   auto c1 = 1.0 / (2 * (1.0 - t));
  //   auto tpow = Vector3<T>{1.0, t, t * t};
  //   auto point = eval(t);
  //   auto row1 = Vector3<T>{point[0], controls[0][0], controls[0][2]};
  //   auto row2 = Vector3<T>{point[1], controls[1][0], controls[1][2]};
  //   auto row3 = Vector3<T>{point[2], controls[2][0], controls[2][2]};

  //   controls[0][1] = c1 * dot(dot(row1, BmatT[0]), tpow);
  //   controls[1][1] = c1 * dot(dot(row2, BmatT[1]), tpow);
  //   controls[2][1] = c1 * dot(dot(row3, BmatT[2]), tpow);
  // }
}

//--------------------------------------------------
// Bezier Curve Evaluation Function
template<typename T>
_hd Vector3<T> eval(const BezierCurve<T> curve, T t) {
  const Vector3<T> tpow = {1.0, t, t * t};
  
  auto px = dot(dot(curve.controls[0], PmatT[0]), tpow);
  auto py = dot(dot(curve.controls[1], PmatT[1]), tpow);
  auto pz = dot(dot(curve.controls[2], PmatT[2]), tpow);

  return {px, py, pz};
};


#endif //CBET_BEZIER_CUH