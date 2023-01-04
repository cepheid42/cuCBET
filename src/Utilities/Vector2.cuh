#ifndef CUCBET_VECTOR2_CUH
#define CUCBET_VECTOR2_CUH

#include <iostream>
#include <cmath>

#define _hd __host__ __device__

template<typename T>
struct Vector2 {
  T e[2];

  Vector2() = default;
  _hd Vector2(T e0, T e1) : e{e0, e1} {};

  _hd T x() const { return e[0]; }
  _hd T y() const { return e[1]; }

  // Subscript Operators
  _hd T operator[] (uint32_t idx) const { return e[idx]; }
  _hd T& operator[] (uint32_t idx) { return e[idx]; }

  // Unary Negation
  _hd Vector2 operator-()  const {
    return {-e[0], -e[1]};
  }

  _hd Vector2& operator+=(const Vector2& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    return *this;
  }

  _hd Vector2& operator-=(const Vector2& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    return *this;
  }

  _hd Vector2& operator*=(const T s) {
    e[0] *= s;
    e[1] *= s;
    return *this;
  }

  _hd Vector2& operator/=(const T s) {
    e[0] /= s;
    e[1] /= s;
    return *this;
  }

  _hd T length_squared() const {
    return (e[0] * e[0]) + (e[1] * e[1]);
  }

  _hd T length() const {
    return sqrt(length_squared());
  }
};

//--------------------------------------------------
// Vector2-Scalar Operations
template<typename T>
_hd Vector2<T> operator*(T s, const Vector2<T>& u)  {
  return {s * u.e[0], s * u.e[1]};
}

template<typename T>
_hd Vector2<T> operator*(const Vector2<T>& u, T s)  {
  return s * u;
}

template<typename T>
_hd Vector2<T> operator/(Vector2<T> u, T s) {
  return (T(1) / s) * u;
}

//--------------------------------------------------
// Vector2-Vector2 Operators
template<typename T>
_hd Vector2<T> operator+(const Vector2<T>& u, const Vector2<T>& v) {
  return {u.e[0] + v.e[0],
          u.e[1] + v.e[1]};
}

template<typename T>
_hd Vector2<T> operator-(const Vector2<T>& u, const Vector2<T>& v) {
  return {u.e[0] - v.e[0],
          u.e[1] - v.e[1]};
}

//--------------------------------------------------
// Vector2 Product Functions
template<typename T>
_hd Vector2<T> unit_vector(Vector2<T> u){
  return u / u.length();
}

template<typename T>
_hd T dot(const Vector2<T>& u, const Vector2<T>& v) {
  // Performs u @ v
  return (u.e[0] * v.e[0]) + (u.e[1] * v.e[1]));
}

template<typename T>
_hd Vector2<T> cross(const Vector2<T>& u, const Vector2<T>& v) {
  // Performs u x v
  return {u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

//--------------------------------------------------
// Printing Function
template<typename T>
std::ostream& operator<<(std::ostream& out, const Vector2<T>& dat) {
  return (out << dat.e[0] << ' ' << dat.e[1]);
}

#endif //CUCBET_VECTOR2_CUH
