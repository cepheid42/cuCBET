#ifndef CUCBET_VEC2_CUH
#define CUCBET_VEC2_CUH

#include <iostream>
#include <cmath>

#define hd __host__ __device__

template<typename T>
struct vec2 {
  T e[2];

  vec2() = default;
  hd vec2(T e0, T e1) : e{e0, e1} {};

  hd T x() const { return e[0]; }
  hd T y() const { return e[1]; }

  // Subscript Operators
  hd       T& operator[] (uint32_t idx)       { return e[idx]; }
  hd const T& operator[] (uint32_t idx) const { return e[idx]; }

  // Unary Negation
  hd vec2 operator-()  const {
    return {-e[0], -e[1]};
  }

  hd vec2& operator+=(const vec2& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    return *this;
  }

  hd vec2& operator-=(const vec2& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    return *this;
  }

  hd vec2& operator*=(const T s) {
    e[0] *= s;
    e[1] *= s;
    return *this;
  }

  hd vec2& operator/=(const T s) {
    e[0] /= s;
    e[1] /= s;
    return *this;
  }

  hd T length_squared() const {
    return (e[0] * e[0]) + (e[1] * e[1]);
  }

  hd T length() const {
    return sqrt(length_squared());
  }
};

//--------------------------------------------------
// vec2-Scalar Operations
template<typename T>
hd vec2<T> operator*(T s, const vec2<T>& u)  {
  return {s * u.e[0], s * u.e[1]};
}

template<typename T>
hd vec2<T> operator*(const vec2<T>& u, T s)  {
  return s * u;
}

template<typename T>
hd vec2<T> operator/(vec2<T> u, T s) {
  return (T(1) / s) * u;
}

//--------------------------------------------------
// vec2-vec2 Operators
template<typename T>
hd vec2<T> operator+(const vec2<T>& u, const vec2<T>& v) {
  return {u.e[0] + v.e[0],
          u.e[1] + v.e[1]};
}

template<typename T>
hd vec2<T> operator-(const vec2<T>& u, const vec2<T>& v) {
  return {u.e[0] - v.e[0],
          u.e[1] - v.e[1]};
}

//--------------------------------------------------
// vec2 Product Functions
template<typename T>
hd vec2<T> unit_vector(vec2<T> u){
  return u / u.length();
}

template<typename T>
hd T dot(const vec2<T>& u, const vec2<T>& v) {
  // Performs u @ v
  return (u.e[0] * v.e[0]) + (u.e[1] * v.e[1]);
}

//--------------------------------------------------
// Printing Function
template<typename T>
std::ostream& operator<<(std::ostream& out, const vec2<T>& dat) {
  return (out << dat.e[0] << ' ' << dat.e[1]);
}

#endif //CUCBET_vec2_CUH
