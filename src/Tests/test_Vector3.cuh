#ifndef CUCBET_TEST_VECTOR3_CUH
#define CUCBET_TEST_VECTOR3_CUH

#include <cassert>
#include <iostream>
#include <cmath>

#include "Vector3.cuh"

using vec3 = Vector3<float>;

#define cudaChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    std::cout << "[" << file << ":" << line << "] GPU Error: " << cudaGetErrorString(code) << std::endl;
    if (abort) { exit(code); }
  }
}

template<typename T>
__global__ void gpu_test_vector3(Vector3<T> x_hat, Vector3<T> y_hat, Vector3<T> z_hat) {
  const float EPSILON = 1.0E-7f;

  // Test default initializer
  vec3 def_init{};
  assert(def_init[0] == 0.0 && def_init[1] == 0.0 && def_init[2] == 0.0);

  // Test pass-by-value and getter methods
  assert(x_hat[0] == 1.0 && x_hat[1] == 0.0 && x_hat[2] == 0.0);
  assert(y_hat[0] == 0.0 && y_hat[1] == 1.0 && y_hat[2] == 0.0);
  assert(z_hat.x() == 0.0 && z_hat.y() == 0.0 && z_hat.z() == 1.0);

  // Test value initialization and length functions
  vec3 xyz1{1.0, 1.0, 1.0};
  assert(xyz1[0] == 1.0 && xyz1[1] == 1.0 && xyz1[2] == 1.0);
  assert(xyz1.length_squared() == 3.0);
  assert(abs(xyz1.length() - sqrt(3.0)) <= EPSILON);

  // Test Unit Vector function and assignment operator
  const float unit_length = 1.0f / sqrt(3.0f);
  vec3 xyz1_unit = unit_vector(xyz1);
  assert(xyz1_unit[0] == unit_length && xyz1_unit[1] == unit_length && xyz1_unit[2] == unit_length);

  // Test Unary Negation
  vec3 xyz1_neg = -xyz1;
  assert(xyz1_neg[0] == -1.0 && xyz1_neg[1] == -1.0 && xyz1_neg[2] == -1.0);

  // Test Augmented Assignment
  vec3 augplus = xyz1;
  augplus += xyz1;
  assert(augplus[0] == 2.0 && augplus[1] == 2.0 && augplus[2] == 2.0);

  vec3 augminus = xyz1;
  augminus -= vec3{2.0, 2.0, 2.0};
  assert(augminus[0] == -1.0 && augminus[1] == -1.0 && augminus[2] == -1.0);

  vec3 augmul = xyz1;
  augmul *= 2.0;
  assert(augmul[0] == 2.0 && augmul[1] == 2.0 && augmul[2] == 2.0);

  vec3 augdiv = augmul;
  augdiv /= 2.0;
  assert(augdiv[0] == 1.0 && augdiv[1] == 1.0 && augdiv[2] == 1.0);

  // Test Vector3-Scalar Operators
  vec3 rmul = xyz1 * 2.0f;
  assert(rmul[0] == 2.0 && rmul[1] == 2.0 && rmul[2] == 2.0);

  vec3 lmul = 2.0f * xyz1;
  assert(lmul[0] == 2.0 && lmul[1] == 2.0 && lmul[2] == 2.0);

  vec3 div = augmul / 2.0f;
  assert(div[0] == 1.0 && div[1] == 1.0 && div[2] == 1.0);

  // Test Vector3-Vector3 Operators
  vec3 vplus = xyz1 + augmul;
  assert(vplus[0] == 3.0 && vplus[1] == 3.0 && vplus[2] == 3.0);

  vec3 vminus = augmul - xyz1;
  assert(vminus[0] == 1.0 && vminus[1] == 1.0 && vminus[2] == 1.0);

  // Test Dot and Cross Product
  assert(dot(x_hat, z_hat) == 0.0);
  assert(dot(augmul, y_hat) == 2.0);

  vec3 x_cross = cross(x_hat, z_hat);
  assert(x_cross[0] == 0.0 && x_cross[1] == -1.0 && x_cross[2] == 0.0);

  // Test Rotations About Axes
  const float pi2 = 3.14159265f / 2.0f;

  vec3 rotx = rotate_x_axis(z_hat, pi2);
  assert(rotx[0] == 0.0 && rotx[1] == -1.0 && abs(rotx[2]) <= EPSILON);

  vec3 roty = rotate_y_axis(z_hat, pi2);
  assert(roty[0] == 1.0 && roty[1] == 0.0 && abs(roty[2]) <= EPSILON);

  vec3 rotz = rotate_z_axis(x_hat, pi2);
  assert(abs(rotz[0]) <= EPSILON && rotz[1] == -1.0 && rotz[2] == 0.0);

  vec3 rotself = rotate_z_axis(z_hat, pi2);
  assert(rotself[0] == 0.0 && rotself[1] == 0.0 && rotself[2] == 1.0);

}

void test_vector3(bool test_gpu) {
  std::cout << "Testing Vector3<float> on CPU." << std::endl;
  const float EPSILON = 1.0E-7f;

  // Test default initializer
  vec3 def_init{};
  assert(def_init[0] == 0.0 && def_init[1] == 0.0 && def_init[2] == 0.0);

  // Test value initialization and getter methods
  vec3 x_hat{1.0, 0.0, 0.0};
  vec3 y_hat{0.0, 1.0, 0.0};
  vec3 z_hat{0.0, 0.0, 1.0};
  assert(x_hat[0] == 1.0 && x_hat[1] == 0.0 && x_hat[2] == 0.0);
  assert(y_hat[0] == 0.0 && y_hat[1] == 1.0 && y_hat[2] == 0.0);
  assert(z_hat.x() == 0.0 && z_hat.y() == 0.0 && z_hat.z() == 1.0);

  // Test length functions
  vec3 xyz1{1.0, 1.0, 1.0};
  assert(xyz1.length_squared() == 3.0);
  assert(abs(xyz1.length() - sqrt(3.0)) <= EPSILON);

  // Test Unit Vector function and assignment operator
  const float unit_length = 1.0f / sqrt(3.0f);
  vec3 xyz1_unit = unit_vector(xyz1);
  assert(xyz1_unit[0] == unit_length && xyz1_unit[1] == unit_length && xyz1_unit[2] == unit_length);

  // Test Unary Negation
  vec3 xyz1_neg = -xyz1;
  assert(xyz1_neg[0] == -1.0 && xyz1_neg[1] == -1.0 && xyz1_neg[2] == -1.0);

  // Test Augmented Assignment
  vec3 augplus = xyz1;
  augplus += xyz1;
  assert(augplus[0] == 2.0 && augplus[1] == 2.0 && augplus[2] == 2.0);

  vec3 augminus = xyz1;
  augminus -= vec3{2.0, 2.0, 2.0};
  assert(augminus[0] == -1.0 && augminus[1] == -1.0 && augminus[2] == -1.0);

  vec3 augmul = xyz1;
  augmul *= 2.0;
  assert(augmul[0] == 2.0 && augmul[1] == 2.0 && augmul[2] == 2.0);

  vec3 augdiv = augmul;
  augdiv /= 2.0;
  assert(augdiv[0] == 1.0 && augdiv[1] == 1.0 && augdiv[2] == 1.0);

  // Test Vector3-Scalar Operators
  vec3 rmul = xyz1 * 2.0f;
  assert(rmul[0] == 2.0 && rmul[1] == 2.0 && rmul[2] == 2.0);

  vec3 lmul = 2.0f * xyz1;
  assert(lmul[0] == 2.0 && lmul[1] == 2.0 && lmul[2] == 2.0);

  vec3 div = augmul / 2.0f;
  assert(div[0] == 1.0 && div[1] == 1.0 && div[2] == 1.0);

  // Test Vector3-Vector3 Operators
  vec3 vplus = xyz1 + augmul;
  assert(vplus[0] == 3.0 && vplus[1] == 3.0 && vplus[2] == 3.0);

  vec3 vminus = augmul - xyz1;
  assert(vminus[0] == 1.0 && vminus[1] == 1.0 && vminus[2] == 1.0);

  // Test Dot and Cross Product
  assert(dot(x_hat, z_hat) == 0.0);
  assert(dot(augmul, y_hat) == 2.0);

  vec3 x_cross = cross(x_hat, z_hat);
  assert(x_cross[0] == 0.0 && x_cross[1] == -1.0 && x_cross[2] == 0.0);

  // Test Rotations About Axes
  const float pi2 = 3.14159265f / 2.0f;

  vec3 rotx = rotate_x_axis(z_hat, pi2);
  assert(rotx[0] == 0.0 && rotx[1] == -1.0 && abs(rotx[2]) <= EPSILON);

  vec3 roty = rotate_y_axis(z_hat, pi2);
  assert(roty[0] == 1.0 && roty[1] == 0.0 && abs(roty[2]) <= EPSILON);

  vec3 rotz = rotate_z_axis(x_hat, pi2);
  assert(abs(rotz[0]) <= EPSILON && rotz[1] == -1.0 && rotz[2] == 0.0);

  vec3 rotself = rotate_z_axis(z_hat, pi2);
  assert(rotself[0] == 0.0 && rotself[1] == 0.0 && rotself[2] == 1.0);

  if (test_gpu) {
    std::cout << "Testing Vector3<float> on GPU." << std::endl;
    gpu_test_vector3<<<1, 1>>>(x_hat, y_hat, z_hat);
    cudaChk(cudaDeviceSynchronize())
  }

  std::cout << "All tests passed." << std::endl;
}

#endif //CUCBET_TEST_VECTOR3_CUH
