#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include <vector>
#include "vec2.cuh"

class Ray {
public:
	Ray() = default;
	Ray(const Point& origin, const Vec& direction) : orig(origin), init_dir(direction) {}

	Point origin() const { return orig; }
	Vec direction() const { return init_dir; }

public:
	Point orig;
	Vec init_dir;
	std::vector<Point> path;
	std::vector<Point> intersections;
};


void draw_init_path(Ray& r, float range, int nt) {
	for (int t = 0; t < nt; ++t) {
		r.path.emplace_back(r.orig + (0.5f * t) * r.init_dir);
	}
}

void ray_intersections(Ray& r1, Ray& r2) {
	for(vec2& p1: r1.path) {
		for (vec2& p2: r2.path) {
			if (p1 == p2) {
				r1.intersections.emplace_back(p1);
			}
		}
	}
}

#endif //CUCBET_RAY_CUH