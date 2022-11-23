#ifndef GPUEM_VECTOR3_CUH
#define GPUEM_VECTOR3_CUH

#include <iostream>
#include <cmath>

#define _hd __host__ __device__

template<typename T>
struct Vector3 {
  T e[3];
  
  Vector3() = default;
  _hd Vector3(T e0, T e1, T e2) : e{e0, e1, e2} {};

  _hd T x() const { return e[0]; }
  _hd T y() const { return e[1]; }
  _hd T z() const { return e[2]; }

  // Subscript Operators
  _hd T operator[] (uint32_t idx) const { return e[idx]; }
  _hd T& operator[] (uint32_t idx) { return e[idx]; }
  
  // Unary Negation
  _hd Vector3 operator-()  const {
    return {-e[0], -e[1], -e[2]};
  }
  
  // Assignment Operators
  _hd Vector3& operator=(const Vector3& rhs) {
    if (this == &rhs) { return *this; }
    e[0] = rhs.e[0];
    e[1] = rhs.e[1];
    e[2] = rhs.e[2];
    return *this;
  }

  _hd Vector3& operator+=(const Vector3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  _hd Vector3& operator-=(const Vector3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
  }

  _hd Vector3& operator*=(const T s) {
    e[0] *= s;
    e[1] *= s;
    e[2] *= s;
    return *this;
  }

  _hd Vector3& operator/=(const T s) {
    e[0] /= s;
    e[1] /= s;
    e[2] /= s;
    return *this;
  }

  _hd T length_squared() const {
    return (e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]);
  }

  _hd T length() const {
    return sqrt(length_squared());
  }
};

//--------------------------------------------------
// Vector3-Scalar Operations
template<typename T>
_hd Vector3<T> operator*(T s, const Vector3<T>& u)  {
  return {s * u.e[0], s * u.e[1], s * u.e[2]};
}

template<typename T>
_hd Vector3<T> operator*(const Vector3<T>& u, T s)  {
  return s * u;
}

template<typename T>
_hd Vector3<T> operator/(Vector3<T> u, T s) {
  return (1 / s) * u;
}

//--------------------------------------------------
// Vector3-Vector3 Operators
template<typename T>
_hd Vector3<T> operator+(const Vector3<T>& u, const Vector3<T>& v) {
  return {u.e[0] + v.e[0],
          u.e[1] + v.e[1],
          u.e[2] + v.e[2]};
}

template<typename T>
_hd Vector3<T> operator-(const Vector3<T>& u, const Vector3<T>& v) {
  return {u.e[0] - v.e[0],
          u.e[1] - v.e[1],
          u.e[2] - v.e[2]};
}

//--------------------------------------------------
// Vector3 Product Functions
template<typename T>
_hd Vector3<T> unit_vector(Vector3<T> u){
  return u / u.length();
}

template<typename T>
_hd T dot(const Vector3<T>& u, const Vector3<T>& v) {
  // Performs u @ v
  return (u.e[0] * v.e[0]) + (u.e[1] * v.e[1]) + (u.e[2] * v.e[2]);
}

template<typename T>
_hd Vector3<T> cross(const Vector3<T>& u, const Vector3<T>& v) {
  // Performs u x v
  return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
          u.e[2] * v.e[0] - u.e[0] * v.e[2],
          u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

//--------------------------------------------------
// Vector3 Coordinate Transforms
/*
 *  Cartesian:   (x, y, z)
 *  Cylindrical: (r, phi, z)      phi angle around Z axis.
 *  Spherical:   (r, theta, phi)  theta angle from +Z axis.
 */

/* To Cartesian Coordinates */
template<typename T>
_hd Vector3<T> cylindrical_to_cartesian(const Vector3<T>& v) {
  auto x = v[0] * cos(v[1]);
  auto y = v[0] * sin(v[1]);
  return {x, y, v[2]};
}

template<typename T>
_hd Vector3<T> spherical_to_cartesian(const Vector3<T>& v) {
  auto x = v[0] * sin(v[2]) * cos(v[1]);
  auto y = v[0] * sin(v[2]) * sin(v[1]);
  auto z = v[0] * cos(v[2]);
  return {x, y, z};
}

/* To Cylindrical Coordinates */
template<typename T>
_hd Vector3<T> cartesian_to_cylindrical(const Vector3<T>& v) {
  auto r = sqrt((v[0] * v[0]) + (v[1] * v[1]));
  auto phi = atan2(v[1], v[2]);
  return {r, phi, v[2]};
}

template<typename T>
_hd Vector3<T> spherical_to_cylindrical(const Vector3<T>& v) {
  auto r = v[0] * sin(v[2]);
  auto z = v[0] * cos(v[2]);
  return {r, v[1], z};
}

/* To Spherical Coordinates */
template<typename T>
_hd Vector3<T> cartesian_to_spherical(const Vector3<T>& v) {
  auto rho = v.length();
  auto theta = acos(v[2] / rho);
  auto phi = acos(v[0] / sqrt((v[0] * v[0]) + (v[1] * v[1])));
  return {rho, theta, phi};
}

template<typename T>
_hd Vector3<T> cylindrical_to_spherical(const Vector3<T>& v) {
  auto rho = sqrt((v[0] * v[0]) * (v[2] * v[2]));
  auto theta = atan2(v[0], v[2]);
  return {rho, theta, v[1]};
}

//--------------------------------------------------
// Vector3 Rotations About Cartesian Axes
template<typename T>
_hd Vector3<T> rotate_x_axis(const Vector3<T>& v, T theta) {
  // rotates Vector3 v around x-axis by theta radians
  auto yp = v[1] * cos(theta) - v[2] * sin(theta);
  auto zp = v[1] * sin(theta) + v[2] * cos(theta);
  return {v[0], yp, zp};
}

// rotates Vector3 v around y-axis by theta radians
template<typename T>
_hd Vector3<T> rotate_y_axis(const Vector3<T>& v, T theta) {
  // rotates Vector3 v around y-axis by theta radians
  auto xp = v[0] * cos(theta) + v[2] * sin(theta);
  auto zp = -v[0] * sin(theta) + v[2] * cos(theta);
  return {xp, v[1], zp};
}

template<typename T>
_hd Vector3<T> rotate_z_axis(const Vector3<T>& v, T theta) {
  // rotates Vector3 v around z-axis by theta radians
  auto xp = v[0] * cos(theta) + v[1] * sin(theta);
  auto yp = -v[0] * sin(theta) + v[1] * cos(theta);
  return {xp, yp, v[2]};
}

//--------------------------------------------------
// Printing Function
template<typename T>
std::ostream& operator<<(std::ostream& out, const Vector3<T>& dat) {
  return (out << dat.e[0] << ' ' << dat.e[1] << ' ' << dat.e[2]);
}

#endif //GPUEM_VECTOR3_CUH
