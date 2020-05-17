#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"

const int ray_len = 100;

class Ray {
public:
	Ray() = default;
	Ray(const Point& origin, const Vec& direction) : orig(origin), init_dir(direction) {}

	Point origin() const { return orig; }
	Vec direction() const { return init_dir; }

public:
	Point orig;
	Vec init_dir;
	Point path[ray_len];
};


// This can be made into a CUDA function for lots of ray drawing quickly... maybe
void draw_init_path(Ray& r) {
	for (int t = 0; t < ray_len; ++t) {
		r.path[t] = r.orig + t * r.init_dir;
	}
}



#endif //CUCBET_RAY_CUH