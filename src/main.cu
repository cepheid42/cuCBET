#include "Utilities/Utilities.cuh"

#include "Beams/Rays2D.cuh"

int main() {

  vec3<float> A{0.0, 0.0, 0.0};
  vec3<float> B{1.0, 1.0, 1.0};
  vec3<float> C{2.0, 2.0, 2.0};
  vec3<float> D{3.0, 3.0, 3.0};

  Ray2D r1{A, B, C, D, 10.0};



  return 0;
}
