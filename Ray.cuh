#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"
#include "Egrid.cuh"

class Ray {
public:
	Ray() = default;
	Ray(const Point& origin, const Vec& direction, float power, int nt) : orig(origin), init_dir(direction), init_power(power) {
		path.reserve(nt);
		group_v.reserve(nt);
		intersections.reserve(nrays);
	}

public:
	Point orig;
	Vec init_dir;
	float init_power{};
	std::vector<Point> path;
	std::vector<Vec> group_v;
	std::vector<int> intersections;
};

bool ray_out_of_range(Point& p) {
	bool valid_x = (p.x() < (xmin - (dx / 2.0f)) || p.x() > (xmax + (dx / 2.0f)));
	bool valid_z = (p.y() < (ymin - (dy / 2.0f)) || p.y() > (ymax + (dy / 2.0f)));
	return (valid_x || valid_z);
}

void draw_init_path(Ray& r, int nt, float dt, float ncrit, Egrid& eg, array<array<float, nx + 2>, ny + 2>& edep, array<array<float, nx>, ny>& present) {
	// This function has loop dependence on t,
	// probably cannot be fully parallelized
	auto ix = get_x_index(r.orig.x(), xmax, xmin, nx);
	auto iy = get_y_index(r.orig.y(), ymax, ymin, ny);

	auto track_x = ix;
	auto track_y = iy;
	present[iy][ix] += 1.0f;

	float wpe = std::sqrt(eg.eden[iy][ix] * 1.0e6f * std::pow(e_c, 2.0f) / (m_e * e_0));

	float k = std::sqrt((std::pow(omega, 2.0f) - std::pow(wpe, 2.0f)) / std::pow(c, 2.0f));
	Vec k1 = k * unit_vector(r.init_dir);
	Vec v1 = (k1 * std::pow(c, 2.0f)) / omega;

	r.path.emplace_back(r.orig);    // Initial position
	r.group_v.emplace_back(v1);     // Initial group velocity

	for (int t = 1; t < nt; t++) {      // Starts from 1, origin is already in path
		auto cur_v = r.group_v[t - 1] - std::pow(c, 2.0f) / (2.0f * ncrit) * eg.d_eden[iy][ix] * dt;
		auto cur_p = r.path[t - 1] + cur_v * dt;

		if (ray_out_of_range(cur_p)) {
			break;
		}

		r.path.emplace_back(cur_p);
		r.group_v.emplace_back(cur_v);

		int x_pos = get_x_index(cur_p.x(), xmax, xmin, nx);
		int y_pos = get_y_index(cur_p.y(), ymax, ymin, ny);

		// Track cells that ray passes through
		if (x_pos != track_x || y_pos != track_y) {
			present[track_y][track_x] += 1.0f;
			track_x = x_pos;
			track_y = y_pos;
		}

		float xp = (cur_p.x() - (get_x_val(x_pos, xmax, xmin, nx) - dx / 2.0f)) / dx;
		float yp = (cur_p.y() - (get_y_val(y_pos, ymax, ymin, ny) - dy / 2.0f)) / dy;

		float dl = std::abs(xp);
		float dm = std::abs(yp);

		// If uray is never updated, increment = init_power
		float a1 = (1.0f - dl) * (1.0f - dm) * r.init_power;
		float a2 = (1.0f - dl) * dm          * r.init_power;
		float a3 = dl * (1.0f - dm)          * r.init_power;
		float a4 = dl * dm                   * r.init_power;

		if (xp >= 0 && yp >= 0) {
			edep[y_pos + 1][x_pos + 1] += a1;
			edep[y_pos + 1][x_pos + 2] += a2;
			edep[y_pos + 2][x_pos + 1] += a3;
			edep[y_pos + 2][x_pos + 2] += a4;
		}
		else if (xp < 0 && yp >= 0) {
			edep[y_pos + 1][x_pos + 1] += a1;
			edep[y_pos + 1][x_pos + 0] += a2;
			edep[y_pos + 2][x_pos + 1] += a3;
			edep[y_pos + 2][x_pos + 0] += a4;
		}
		else if (xp >= 0 && yp < 0) {
			edep[y_pos + 1][x_pos + 1] += a1;
			edep[y_pos + 1][x_pos + 2] += a2;
			edep[y_pos + 0][x_pos + 1] += a3;
			edep[y_pos + 0][x_pos + 2] += a4;
		}
		else if (xp < 0 && yp < 0) {
			edep[y_pos + 1][x_pos + 1] += a1;
			edep[y_pos + 1][x_pos + 0] += a2;
			edep[y_pos + 0][x_pos + 1] += a3;
			edep[y_pos + 0][x_pos + 0] += a4;
		}
		else {
			std::cout << "Error in deposition grid interpolation." << std::endl;
			assert(0);
		}
	}
}

#endif //CUCBET_RAY_CUH