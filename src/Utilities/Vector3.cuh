#ifndef CUCBET_VEC3_CUH
#define CUCBET_VEC3_CUH

#include <iostream>
#include <cmath>

#define hd __host__ __device__

template<typename T>
struct vec3 {
  T e[3];
  
  vec3() = default;
  hd vec3(T e0, T e1, T e2) : e{e0, e1, e2} {};

  hd T x() const { return e[0]; }
  hd T y() const { return e[1]; }
  hd T z() const { return e[2]; }
  hd T w() const { return e[2]; }

  // Subscript Operators
  hd       T& operator[] (size_t idx)       { return e[idx]; }
  hd const T& operator[] (size_t idx) const { return e[idx]; }
  
  // Unary Negation
  hd vec3 operator-()  const {
    return {-e[0], -e[1], -e[2]};
  }

  hd vec3& operator+=(const vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  hd vec3& operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
  }

  hd vec3& operator*=(const T s) {
    e[0] *= s;
    e[1] *= s;
    e[2] *= s;
    return *this;
  }

  hd vec3& operator/=(const T s) {
    e[0] /= s;
    e[1] /= s;
    e[2] /= s;
    return *this;
  }

  hd T length_squared() const {
    return (e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]);
  }

  hd T length() const {
    return sqrt(length_squared());
  }
};

//--------------------------------------------------
// vec3-Scalar Operations
template<typename T>
hd vec3<T> operator*(T s, const vec3<T>& u)  {
  return {s * u.e[0], s * u.e[1], s * u.e[2]};
}

template<typename T>
hd vec3<T> operator*(const vec3<T>& u, T s)  {
  return s * u;
}

template<typename T>
hd vec3<T> operator/(vec3<T> u, T s) {
  return (1 / s) * u;
}

//--------------------------------------------------
// vec3-vec3 Operators
template<typename T>
hd vec3<T> operator+(const vec3<T>& u, const vec3<T>& v) {
  return {u.e[0] + v.e[0],
          u.e[1] + v.e[1],
          u.e[2] + v.e[2]};
}

template<typename T>
hd vec3<T> operator-(const vec3<T>& u, const vec3<T>& v) {
  return {u.e[0] - v.e[0],
          u.e[1] - v.e[1],
          u.e[2] - v.e[2]};
}

//--------------------------------------------------
// vec3 Product Functions
template<typename T>
hd vec3<T> unit_vector(vec3<T> u){
  return u / u.length();
}

template<typename T>
hd T dot(const vec3<T>& u, const vec3<T>& v) {
  // Performs u @ v
  return (u.e[0] * v.e[0]) + (u.e[1] * v.e[1]) + (u.e[2] * v.e[2]);
}

template<typename T>
hd vec3<T> cross(const vec3<T>& u, const vec3<T>& v) {
  // Performs u x v
  return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
          u.e[2] * v.e[0] - u.e[0] * v.e[2],
          u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

//--------------------------------------------------
// vec3 Coordinate Transforms
/*
 *  Cartesian:   (x, y, z)
 *  Cylindrical: (r, phi, z)      phi angle around Z axis.
 *  Spherical:   (r, theta, phi)  theta angle from +Z axis.
 */

/* To Cartesian Coordinates */
template<typename T>
hd vec3<T> cylindrical_to_cartesian(const vec3<T>& v) {
  auto x = v[0] * cos(v[1]);
  auto y = v[0] * sin(v[1]);
  return {x, y, v[2]};
}

template<typename T>
hd vec3<T> spherical_to_cartesian(const vec3<T>& v) {
  auto x = v[0] * sin(v[2]) * cos(v[1]);
  auto y = v[0] * sin(v[2]) * sin(v[1]);
  auto z = v[0] * cos(v[2]);
  return {x, y, z};
}

/* To Cylindrical Coordinates */
template<typename T>
hd vec3<T> cartesian_to_cylindrical(const vec3<T>& v) {
  auto r = sqrt((v[0] * v[0]) + (v[1] * v[1]));
  auto phi = atan2(v[1], v[2]);
  return {r, phi, v[2]};
}

template<typename T>
hd vec3<T> spherical_to_cylindrical(const vec3<T>& v) {
  auto r = v[0] * sin(v[2]);
  auto z = v[0] * cos(v[2]);
  return {r, v[1], z};
}

/* To Spherical Coordinates */
template<typename T>
hd vec3<T> cartesian_to_spherical(const vec3<T>& v) {
  auto rho = v.length();
  auto theta = acos(v[2] / rho);
  auto phi = acos(v[0] / sqrt((v[0] * v[0]) + (v[1] * v[1])));
  return {rho, theta, phi};
}

template<typename T>
hd vec3<T> cylindrical_to_spherical(const vec3<T>& v) {
  auto rho = sqrt((v[0] * v[0]) * (v[2] * v[2]));
  auto theta = atan2(v[0], v[2]);
  return {rho, theta, v[1]};
}

//--------------------------------------------------
// vec3 Rotations About Cartesian Axes
template<typename T>
hd vec3<T> rotate_x_axis(const vec3<T>& v, T theta) {
  // rotates vec3 v around x-axis by theta radians
  auto yp = v[1] * cos(theta) - v[2] * sin(theta);
  auto zp = v[1] * sin(theta) + v[2] * cos(theta);
  return {v[0], yp, zp};
}

// rotates vec3 v around y-axis by theta radians
template<typename T>
hd vec3<T> rotate_y_axis(const vec3<T>& v, T theta) {
  // rotates vec3 v around y-axis by theta radians
  auto xp = v[0] * cos(theta) + v[2] * sin(theta);
  auto zp = -v[0] * sin(theta) + v[2] * cos(theta);
  return {xp, v[1], zp};
}

template<typename T>
hd vec3<T> rotate_z_axis(const vec3<T>& v, T theta) {
  // rotates vec3 v around z-axis by theta radians
  auto xp = v[0] * cos(theta) + v[1] * sin(theta);
  auto yp = -v[0] * sin(theta) + v[1] * cos(theta);
  return {xp, yp, v[2]};
}

//--------------------------------------------------
// Printing Function
template<typename T>
std::ostream& operator<<(std::ostream& out, const vec3<T>& dat) {
  return (out << dat.e[0] << ' ' << dat.e[1] << ' ' << dat.e[2]);
}

#endif //CUCBET_VEC3_CUH
