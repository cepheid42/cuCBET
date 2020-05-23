#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"
#include "Egrid.cuh"

class Ray {
public:
	Ray() = default;
	Ray(const Point& origin, const Vec& direction, double power, int nt) : orig(origin), init_dir(direction), init_power(power) {
		path.reserve(nt);
		group_v.reserve(nt);
	}

public:
	Point orig;
	Vec init_dir;
	double init_power{};
	std::vector<Point> path;
	std::vector<Vec> group_v;
	std::vector<Point*> intersections;
};

void find_intersections(Ray& r1, Ray& r2) {
	for (auto& p1: r1.path) {
		for (auto& p2: r2.path) {
			if (p1 == p2) {
				r1.intersections.emplace_back(&p1);
			}
		}
	}
}

bool ray_out_of_range(Point& p) {
	bool valid_x = (p.x() < (xmin - (dx / 2.0)) || p.x() > (xmax + (dx / 2.0)));
	bool valid_z = (p.z() < (zmin - (dz / 2.0)) || p.z() > (zmax + (dz / 2.0)));
	return (valid_x || valid_z);
}

int get_point_path_index(Ray& ray, Point& p1) {
	for (int i = 0; i < ray.path.size(); i++) {
		if (p1 == ray.path[i]) {
			return i;
		}
	}
	return -1;
}

Point* get_previous_point(Ray& ray, Point& p1) {
	if (p1 == ray.path[0]){
		return &p1;
	}
	for (int i = 0; i < ray.path.size(); i++) {
		if (p1 == ray.path[i]) {
			return &(ray.path[i - 1]);
		}
	}
	return nullptr;
}

void draw_init_path(Ray& r, int nt, double dt, double ncrit, Egrid& eg, std::array<std::array<double, nx + 2>, nz + 2>& edep) {
	// This function has loop dependence on t,
	// probably cannot be fully parallelized
	auto x_start = get_index(r.orig[0], xmax, xmin, nx);
	auto z_start = get_index(r.orig[1], zmax, zmin, nz);

	double wpe = sqrt(eg.eden[x_start][z_start] * 1.0e6 * pow(e_c, 2.0) / (m_e * e_0));

	double k = std::sqrt((pow(omega, 2) - pow(wpe, 2)) / pow(c, 2));
	auto k1_xz = k * r.init_dir;

	r.path.emplace_back(r.orig);
	r.group_v.emplace_back((k1_xz * pow(c, 2)) / omega);     // Initial group velocity

	for (int t = 1; t < nt; ++t) {      // Starts from 1, origin is already in path
		auto cur_v = r.group_v[t - 1] - pow(c, 2) / (2.0 * ncrit) * eg.d_eden[x_start][z_start] * dt;
		auto cur_p = r.path[t - 1] + cur_v * dt;

		if (ray_out_of_range(cur_p)) {
			// Ray is out of range
			break;
		}
		r.path.emplace_back(cur_p);
		r.group_v.emplace_back(cur_v);

		int x_pos = get_index(cur_p.x(), xmax, xmin, nx);
		int z_pos = get_index(cur_p.z(), zmax, zmin, nz);

		double xp = (cur_p.x() - (get_grid_val(x_pos, xmax, xmin, nx) + dx / 2.0)) / dx;
		double zp = (cur_p.z() - (get_grid_val(z_pos, zmax, zmin, nz) + dz / 2.0)) / dz;

		double dl = abs(xp);
		double dm = abs(xp);

		double a1 = (1.0 - dl) * (1.0 - dm);
		double a2 = (1.0 - dl) * dm;
		double a3 = dl * (1.0 - dm);
		double a4 = dl * dm;

		if (xp >= 0 && zp >= 0) {
			edep[z_pos + 1][x_pos + 1] += a1 * r.init_power;     // If uray is never updated, increment = init_power
			edep[z_pos + 1][x_pos + 2] += a2 * r.init_power;
			edep[z_pos + 2][x_pos + 1] += a3 * r.init_power;
			edep[z_pos + 2][x_pos + 2] += a4 * r.init_power;
		}
		else if (xp < 0 && zp >= 0) {
			edep[z_pos + 1][x_pos + 1] += a1 * r.init_power;
			edep[z_pos + 1][x_pos + 0] += a2 * r.init_power;
			edep[z_pos + 2][x_pos + 1] += a3 * r.init_power;
			edep[z_pos + 2][x_pos + 0] += a4 * r.init_power;
		}
		else if (xp >= 0 && zp < 0) {
			edep[z_pos + 1][x_pos + 1] += a1 * r.init_power;
			edep[z_pos + 1][x_pos + 2] += a2 * r.init_power;
			edep[z_pos + 0][x_pos + 1] += a3 * r.init_power;
			edep[z_pos + 0][x_pos + 2] += a4 * r.init_power;
		}
		else if (xp < 0 && zp < 0) {
			edep[z_pos + 1][x_pos + 1] += a1 * r.init_power;
			edep[z_pos + 1][x_pos + 0] += a2 * r.init_power;
			edep[z_pos + 0][x_pos + 1] += a3 * r.init_power;
			edep[z_pos + 0][x_pos + 0] += a4 * r.init_power;
		}
		else {
			std::cout << "Error in deposition grid interpolation." << std::endl;
			assert(0);
		}
	}
}

#endif //CUCBET_RAY_CUH