#ifndef CUCBET_TEST_vec2_CUH
#define CUCBET_TEST_vec2_CUH

#include <cassert>
#include <iostream>
#include <cmath>

#include "Vector2.cuh"

using vec2f = vec2<float>;

#define cudaChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    std::cout << "[" << file << ":" << line << "] GPU Error: " << cudaGetErrorString(code) << std::endl;
    if (abort) { exit(code); }
  }
}

template<typename T>
__global__ void gpu_test_vec2(vec2<T> x_hat, vec2<T> y_hat) {
  const float EPSILON = 1.0E-7f;

  // Test default initializer
  vec2f def_init{};
  assert(def_init[0] == 0.0 && def_init[1] == 0.0);

  // Test pass-by-value and getter methods
  assert(x_hat[0] == 1.0 && x_hat[1] == 0.0);
  assert(y_hat[0] == 0.0 && y_hat[1] == 1.0);
  assert(z_hat.x() == 0.0 && z_hat.y() == 0.0);

  // Test value initialization and length functions
  vec2f xy1{1.0, 1.0};
  assert(xy1[0] == 1.0 && xy1[1] == 1.0);
  assert(xy1.length_squared() == 2.0);
  assert(abs(xy1.length() - sqrt(2.0)) <= EPSILON);

  // Test Unit Vector function and assignment operator
  const float unit_length = 1.0f / sqrt(2.0f);
  vec2f xy1_unit = unit_vector(xy1);
  assert(xy1_unit[0] == unit_length && xy1_unit[1] == unit_length);

  // Test Unary Negation
  vec2f xy1_neg = -xy1;
  assert(xy1_neg[0] == -1.0 && xy1_neg[1] == -1.0);

  // Test Augmented Assignment
  vec2f augplus = -xy1_neg;
  augplus += xy1;
  assert(augplus[0] == 2.0 && augplus[1] == 2.0);

  vec2f augminus = -xy1_neg;
  augminus -= vec2f{2.0, 2.0, 2.0};
  assert(augminus[0] == -1.0 && augminus[1] == -1.0);

  vec2f augmul = -xy1_neg;
  augmul *= 2.0;
  assert(augmul[0] == 2.0 && augmul[1] == 2.0);

  // Test default copy constructor
  vec2f augdiv = augmul;
  augdiv /= 2.0;
  assert(augdiv[0] == 1.0 && augdiv[1] == 1.0);

  // Test vec2-Scalar Operators
  vec2f rmul = xy1 * 2.0f;
  assert(rmul[0] == 2.0 && rmul[1] == 2.0);

  vec2f lmul = 2.0f * xy1;
  assert(lmul[0] == 2.0 && lmul[1] == 2.0);

  vec2f div = augmul / 2.0f;
  assert(div[0] == 1.0 && div[1] == 1.0);

  // Test vec2-vec2 Operators
  vec2f vplus = xy1 + augmul;
  assert(vplus[0] == 3.0 && vplus[1] == 3.0);

  vec2f vminus = augmul - xy1;
  assert(vminus[0] == 1.0 && vminus[1] == 1.0);

  // Test Dot and Cross Product
  assert(dot(x_hat, y_hat) == 0.0);
  assert(dot(augmul, y_hat) == 2.0);
}

void test_vec2(bool test_gpu) {
  std::cout << "Testing vec2<float> on CPU." << std::endl;
  const float EPSILON = 1.0E-7f;

  // Test Plain-old-data/Standard-layout
  assert(std::is_standard_layout<vec2<float>>::value == true);

  // Test default initializer
  vec2f def_init{};
  assert(def_init[0] == 0.0 && def_init[1] == 0.0 && def_init[2] == 0.0);

  // Test value initialization and getter methods
  vec2f x_hat{1.0, 0.0};
  vec2f y_hat{0.0, 1.0};
  assert(x_hat[0] == 1.0 && x_hat[1]);
  assert(y_hat[0] == 0.0 && y_hat[1] == 1.0);

  // Test length functions
  vec2f xy1{1.0, 1.0};
  assert(xy1.length_squared() == 2.0);
  assert(abs(xy1.length() - sqrt(2.0)) <= EPSILON);

  // Test Unit Vector function and assignment operator
  const float unit_length = 1.0f / sqrt(2.0f);
  vec2f xy1_unit = unit_vector(xy1);
  assert(xy1_unit[0] == unit_length && xy1_unit[1] == unit_length);

  // Test Unary Negation
  vec2f xy1_neg = -xy1;
  assert(xy1_neg[0] == -1.0 && xy1_neg[1] == -1.0);

  // Test Augmented Assignment
  vec2f augplus = -xy1_neg;
  augplus += xy1;
  assert(augplus[0] == 2.0 && augplus[1] == 2.0);

  vec2f augminus = -xy1_neg;
  augminus -= vec2f{2.0, 2.0};
  assert(augminus[0] == -1.0 && augminus[1] == -1.0);

  vec2f augmul = -xy1_neg;
  augmul *= 2.0;
  assert(augmul[0] == 2.0 && augmul[1] == 2.0);

  // Test default copy constructor
  vec2f augdiv = augmul;
  augdiv /= 2.0;
  assert(augdiv[0] == 1.0 && augdiv[1] == 1.0);

  // Test vec2-Scalar Operators
  vec2f rmul = xy1 * 2.0f;
  assert(rmul[0] == 2.0 && rmul[1] == 2.0);

  vec2f lmul = 2.0f * xy1;
  assert(lmul[0] == 2.0 && lmul[1] == 2.0);

  vec2f div = augmul / 2.0f;
  assert(div[0] == 1.0 && div[1] == 1.0);

  // Test vec2-vec2 Operators
  vec2f vplus = xy1 + augmul;
  assert(vplus[0] == 3.0 && vplus[1] == 3.0);

  vec2f vminus = augmul - xy1;
  assert(vminus[0] == 1.0 && vminus[1] == 1.0);

  // Test Dot and Cross Product
  assert(dot(x_hat, y_hat) == 0.0);
  assert(dot(augmul, y_hat) == 2.0);

  if (test_gpu) {
    std::cout << "Testing vec2<float> on GPU." << std::endl;
    gpu_test_vec2<<<1, 1>>>(x_hat, y_hat);
    cudaChk(cudaDeviceSynchronize())
  }

  std::cout << "All tests passed." << std::endl;
}

#endif //CUCBET_TEST_vec2_CUH
