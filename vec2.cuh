#ifndef CUCBET_vec2_CUH
#define CUCBET_vec2_CUH

#include "constants.cuh"


class vec2 {
public:
	vec2() : e{0,0} {}
	vec2(float e0, float e1) : e{e0, e1} {}
	vec2(const vec2& v) : e{v.e[0], v.e[1]} {}

	vec2& operator=(const vec2& rhs) {
		if (this != &rhs) {
			this->e[0] = rhs.e[0];
			this->e[1] = rhs.e[1];
		}
		return *this;
	}

	float x() const { return e[0]; }
	float y() const { return e[1]; }

	vec2 operator-() const { return vec2(-e[0], -e[1]); }
	float operator[](int i) const { return e[i]; }
	float& operator[](int i) { return e[i]; }

	vec2& operator+=(const vec2 &v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		return *this;
	}

	vec2& operator*=(const float t) {
		e[0] *= t;
		e[1] *= t;
		return *this;
	}

	vec2& operator/=(const float t) {
		return *this *= 1/t;
	}

	float length() const {
		return std::sqrt(length_squared());
	}

	float length_squared() const {
		return e[0]*e[0] + e[1]*e[1];
	}

public:
	float e[2]{};
};

// Aliases
using Point = vec2;
using Vec = vec2;

// vec2 Utility Functions
inline std::ostream& operator<<(std::ostream &out, const vec2 &v) {
	return out << v.e[0] << ", " << v.e[1];
}

inline vec2 operator+(const vec2 &u, const vec2 &v) {
	return vec2(u.e[0] + v.e[0], u.e[1] + v.e[1]);
}

inline vec2 operator-(const vec2 &u, const vec2 &v) {
	return vec2(u.e[0] - v.e[0], u.e[1] - v.e[1]);
}

inline vec2 operator*(const vec2 &u, const vec2 &v) {
	return vec2(u.e[0] * v.e[0], u.e[1] * v.e[1]);
}

inline vec2 operator*(float t, const vec2 &v) {
	return vec2(t*v.e[0], t*v.e[1]);
}

inline vec2 operator*(const vec2& v, float t) {
	return t * v;
}

inline vec2 operator/(const vec2& v, float t) {
	return (1/t) * v;
}

inline vec2 unit_vector(const vec2& v) {
	return v / v.length();
}

#endif //CUCBET_vec2_CUH
