#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"

class Ray {
public:
	Ray() = default;
	Ray(const Point& origin, const Vec& direction) : orig(origin), init_dir(direction) {
		path.emplace_back(orig);
	}

	Point origin() const { return orig; }
	Vec direction() const { return init_dir; }

public:
	double wpe = 1.6970185e15;
	Point orig;
	Vec init_dir;
	std::vector<Point> path;
	std::vector<Point> velocities;
	std::vector<double> intensities;

};


void draw_init_path(Ray& r, int nt, double dt, double ncrit) {
	Point d_eden(1.809914e24, 0.0);
	double k = std::sqrt((pow(omega, 2) - pow(r.wpe, 2)) / pow(c, 2));
	auto k1_xz = k * r.init_dir;

	r.velocities.emplace_back((k1_xz * pow(c, 2)) / omega);

	for (int t = 1; t < nt; ++t) {
		auto next_v = r.velocities[t - 1] - pow(c, 2) / (2.0 * ncrit) * d_eden * dt;
		auto next_p = r.path[t - 1] + next_v * dt;

		r.path.emplace_back(next_p);
		r.velocities.emplace_back(next_v);
	}
}

//void ray_intersections(Ray& r1, Ray& r2) {
//	for(vec2& p1: r1.path) {
//		for (vec2& p2: r2.path) {
//			if (p1 == p2) {
//				r1.intersections.emplace_back(p1);
//			}
//		}
//	}
//}

#endif //CUCBET_RAY_CUH