#ifndef CUCBET_vec2_CUH
#define CUCBET_vec2_CUH

#include "constants.cuh"

class vec2 {
public:
	// Constructors
	__host__ __device__ vec2() : x(0), y(0) {}
	__host__ __device__ vec2(float e0, float e1) : x(e0), y(e1) {}
	vec2(const vec2& v) = default;
	~vec2() = default;

	__host__ __device__ vec2& operator=(const vec2& v) {
		if (&v == this) {
			return *this;
		}
		x = v.x;
		y = v.y;
		return *this;
	}

	__host__ __device__ vec2 operator-() const { return {-x, -y}; }
	__host__ __device__ vec2& operator*=(const float t) {
		x *= t;
		y *= t;
		return *this;
	}

	__host__ __device__ vec2& operator/=(const float t) {
		return *this *= 1/t;
	}

	// Device only for simpler code
	__device__ inline float len() const {
		return sqrtf(x*x + y*y);
	}

	__host__ __device__ inline float sqr_len() const {
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

__host__ __device__ inline vec2 operator+(const vec2 &u, const vec2 &v) {
	return {u.x + v.x, u.y + v.y};
}

__host__ __device__ inline vec2 operator-(const vec2 &u, const vec2 &v) {
	return {u.x - v.x, u.y - v.y};
}

__host__ __device__ inline vec2 operator*(const vec2 &u, const vec2 &v) {
	return {u.x * v.x, u.y * v.y};
}

__host__ __device__ inline vec2 operator*(float t, const vec2 &v) {
	return {t*v.x, t*v.y};
}

__host__ __device__ inline vec2 operator*(const vec2& v, float t) {
	return t * v;
}

__host__ __device__ inline vec2 operator/(const vec2& v, float t) {
	return (1/t) * v;
}

__host__ inline vec2 unit_vector(const vec2& v) {
	return v / std::sqrt(v.x*v.x + v.y*v.y);
}

#endif //CUCBET_vec2_CUH
