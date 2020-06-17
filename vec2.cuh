#ifndef CUCBET_vec2_CUH
#define CUCBET_vec2_CUH

#include "constants.cuh"

class vec2 {
public:
	__hd__ vec2() : x(0), y(0) {}
	__hd__ vec2(float e0, float e1) : x(e0), y(e1) {}
	vec2(const vec2& v) = default;

	__hd__ float getx() const { return x; }
	__hd__ float gety() const { return y; }

	__hd__ vec2 operator-() const { return vec2(-x, -y); }

	__hd__ vec2& operator*=(const float t) {
		x *= t;
		y *= t;
		return *this;
	}

	__hd__ vec2& operator/=(const float t) {
		return *this *= 1/t;
	}

	__hd__ inline float length() const {
		#if __CUDA_ARCH__
			return sqrtf(length_squared());
		#elif !defined(__CUDA_ARCH__)
			return std::sqrt(length_squared());
		#endif
	}

	__hd__ inline float length_squared() const {
		return x*x + y*y;
	}

public:
	float x, y;
};

// Aliases
using Point = vec2;
using Vec = vec2;


// vec2 Utility Functions
inline std::ostream& operator<<(std::ostream &out, const vec2 &v) {
	return out << v.x << ", " << v.y;
}

__hd__ inline vec2 operator+(const vec2 &u, const vec2 &v) {
	return vec2(u.x + v.x, u.y + v.y);
}

__hd__ inline vec2 operator-(const vec2 &u, const vec2 &v) {
	return vec2(u.x - v.x, u.y - v.y);
}

__hd__ inline vec2 operator*(const vec2 &u, const vec2 &v) {
	return vec2(u.x * v.x, u.y * v.y);
}

__hd__ inline vec2 operator*(float t, const vec2 &v) {
	return vec2(t*v.x, t*v.y);
}

__hd__ inline vec2 operator*(const vec2& v, float t) {
	return t * v;
}

__hd__ inline vec2 operator/(const vec2& v, float t) {
	return (1/t) * v;
}

__hd__ inline vec2 unit_vector(const vec2& v) {
	return v / v.length();
}

#endif //CUCBET_vec2_CUH
