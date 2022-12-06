#ifndef CBET_BEZIER_CUH
#define CBET_BEZIER_CUH

#include "../Utilities/Vector3.cuh"

template<typename T>
struct BezierCurve {
  vector<T> controls[3];
  vector<T> mcurve[3] = {Vector<T>(1.0, -2.0, 1.0), Vector<T>(0.0, 2.0, -2.0), Vector3<T>(0.0, 0.0, 1.0)};

  _hd BezierCurve(Vector3<T> A, Vector3<T> B, Vector3<T> C) 
  : controls{Vector3<T>(A[0], B[0], C[0]), Vector3<T>(A[1], B[1], C[1]), Vector3<T>(A[2], B[2], C[2])} 
  {};

  _hd inline void update_curve() {
    #pragma unroll 3
    for(auto i = 0; i < 3; i++) {
      mcurve[i] = dot(mcurve[i], controls[0]);
      mcurve[i + 1] = dot(mcurve[i], controls[1]);
      mcurve[i + 2] = dot(mcurve[i], controls[2]);
    }
  }

  _hd void update_control_c(Vector3<T> new_c) {
    controls[0][2] = new_c[0];
    controls[1][2] = new_c[1];
    controls[2][2] = new_c[2];

    update_curve();
  }

  // Vector3<T> eval(T t) const {
  //   auto a = 1.0 - t;
  //   return (a * a * controls[0]) + (2.0 * a * t * controls[1]) + (t * t * controls[2]);
  // };

  Vector3<T> eval(T t) const {
    Vector3<T> powers{1.0, t, t * t};

    
  }


  explicit operator bool() { /*????*/ };
}

#endif //CBET_BEZIER_CUH