#ifndef GPUEM_VECTOR_CUH
#define GPUEM_VECTOR_CUH

#include <cstdint>
#include <iostream>
#include "Defines.cuh"

template<typename T>
struct Vector {
  T e[3];
  
  _hd Vector() = default;
  _hd Vector(T e0, T e1, T e2) : e{e0, e1, e2} {};

  _hd T x() const { return e[0]; }
  _hd T y() const { return e[1]; }
  _hd T z() const { return e[2]; }

  // Subscript Operators
  _hd T operator[] (uint32_t idx) const { return e[idx]; }
  _hd T& operator[] (uint32_t idx) { return e[idx]; }
  
  // Unary Negation
  _hd Vector operator-()  const {
    return {-e[0], -e[1], -e[2]};
  }
  
  // Assignment Operators
  _hd Vector& operator=(const Vector& rhs) {
    if (this ==& rhs) { return *this; }
    e[0] = rhs.e[0];
    e[1] = rhs.e[1];
    e[2] = rhs.e[2];
    return *this;
  }

  _hd Vector& operator+=(const Vector& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  _hd Vector& operator-=(const Vector& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
  }

  _hd Vector& operator*=(const T s) {
    e[0] *= s;
    e[1] *= s;
    e[2] *= s;
    return *this;
  }

  _hd Vector& operator/=(const T s) {
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
// Vector-Scalar Operations
template<typename T>
_hd Vector<T> operator*(T s, const Vector<T>& u)  {
  return {s * u.e[0], s * u.e[1], s * u.e[2]};
}

template<typename T>
_hd Vector<T> operator*(const Vector<T>& u, T s)  {
  return s * u;
}

template<typename T>
_hd Vector<T> operator/(Vector<T> u, T s) {
  return (1 / s) * u;
}

//--------------------------------------------------
// Vector-Vector Operators
template<typename T>
_hd Vector<T> operator+(const Vector<T>& u, const Vector<T>& v) {
  return {u.e[0] + v.e[0],
          u.e[1] + v.e[1],
          u.e[2] + v.e[2]};
}

template<typename T>
_hd Vector<T> operator-(const Vector<T>& u, const Vector<T>& v) {
  return {u.e[0] - v.e[0],
          u.e[1] - v.e[1],
          u.e[2] - v.e[2]};
}

//--------------------------------------------------
// Vector Utility Methods
 template<typename T>
_hd Vector<T> unit_vector(Vector<T> u){
  return u / u.length();
}

template<typename T>
_hd T dot(const Vector<T>& u, const Vector<T>& v) {
  // Performs u @ v
  return (u.e[0] * v.e[0]) + (u.e[1] * v.e[1]) + (u.e[2] * v.e[2]);
}

template<typename T>
_hd Vector<T> cross(const Vector<T>& u, const Vector<T>& v) {
  // Performs u x v
  return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
          u.e[2] * v.e[0] - u.e[0] * v.e[2],
          u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

// To Cartesian Coordinates
template<typename T>
_hd Vector<T> cylindrical_to_cartesian(const Vector<T>& v) {
  auto x = v[0] * cos(v[1]);
  auto y = v[0] * sin(v[1]);
  return {x, y, v[2]};
}

template<typename T>
_hd Vector<T> spherical_to_cartesian(const Vector<T>& v) {
  auto x = v[0] * sin(v[2]) * cos(v[1]);
  auto y = v[0] * sin(v[2]) * sin(v[1]);
  auto z = v[0] * cos(v[2]);
  return {x, y, z};
}

// To Cylindrical Coordinates
template<typename T>
_hd Vector<T> cartesian_to_cylindrical(const Vector<T>& v) {
  auto r = sqrt((v[0] * v[0]) + (v[1] * v[1]));
  auto theta = atan2(v[1], v[2]);
  return {r, theta, v[2]};
}

// To Spherical Coordinates
template<typename T>
_hd Vector<T> cartesian_to_spherical(const Vector<T>& v) {
  auto rho = v.length();
  auto theta = acos(v[2] / rho);
  auto phi = acos(v[0] / sqrt((v[0] * v[0]) + (v[1] * v[1])));
  return {rho, theta, phi};
}

// rotates vector v around z-axis by theta radians
template<typename T>
_hd Vector<T> rotate(const Vector<T>& v, T theta) {
  auto xp = v[0] * cos(theta) + v[1] * sin(theta);
  auto yp = -v[0] * sin(theta) + v[1] * cos(theta);
  return {xp, yp, v[2]};
}

//--------------------------------------------------
// Printing Function
template<typename T>
std::ostream& operator<<(std::ostream& out, const Vector<T>& dat) {
  return (out << dat.e[0] << ' ' << dat.e[1] << ' ' << dat.e[2]);
}

#endif //GPUEM_VECTOR_CUH
