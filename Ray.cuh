#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"
#include "Egrid.cuh"

class Ray {
public:
	Ray() = default;
	Ray(const Point& origin, const Vec& direction) : orig(origin), init_dir(direction) {
		path.emplace_back(orig);
	}

public:
	double wpe = 1.6970185e15;
	Point orig;
	Vec init_dir;
	std::vector<Point> path;
	std::vector<Vec> velocities;
};


void draw_init_path(Ray& r, int nt, double dt, double ncrit, Egrid& e) {
	// This function has loop dependence on t,
	// probably cannot be fully parallelized

	double k = std::sqrt((pow(omega, 2) - pow(r.wpe, 2)) / pow(c, 2));
	auto k1_xz = k * r.init_dir;    // init_dir is already a unit vector

	r.velocities.emplace_back((k1_xz * pow(c, 2)) / omega);

	for (int t = 1; t < nt; ++t) {
		auto x_start = r.orig[0];
		auto z_start = r.orig[1];
		auto next_v = r.velocities[t - 1] - pow(c, 2) / (2.0 * ncrit) * e.d_eden[x_start][z_start] * dt;
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