#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"
#include "Egrid.cuh"

class Ray {
public:
	Ray() : orig(), dir(), power(0.0f), endex(0), path(nullptr), group_v(nullptr), intersections(nullptr) {}
	~Ray() = default;

	void allocate(const float x_start, const float y_start, const Vec& ndir, float uray0, int nt) {
		orig.x = x_start;
		orig.y = y_start;
		dir = ndir;
		power = uray0;
		endex = 0;

		checkErr(cudaMallocManaged(&path,          nt    * sizeof(Point)))
		checkErr(cudaMallocManaged(&group_v,       nt    * sizeof(Vec)))
		checkErr(cudaMallocManaged(&intersections, nrays * sizeof(Point)))
		checkErr(cudaDeviceSynchronize())
	}

	void free() const {
		checkErr(cudaDeviceSynchronize())
		checkErr(cudaFree(path))
		checkErr(cudaFree(group_v))
		checkErr(cudaFree(intersections))
	}

	void append_path(Point& p, Vec& v) {
		path[endex] = p;
		group_v[endex] = v;
		endex++;
	}

public:
	Point orig;
	Vec dir;
	float power;
	int endex;

	Point* path;
	Vec* group_v;
	Point* intersections;
};

// Utility Functions
inline bool ray_out_of_range(Point& p) {
	return (p.x < xmin - (dx / 2.0f) || p.x > xmax - (dx / 2.0f) || p.y < ymin - (dy / 2) || p.y > ymax - (dy / 2));
}

void draw_init_path(Ray& r, Egrid& eg, float* edep, const int nt) {
	auto ix = get_x_index(r.orig.x, xmax, xmin, nx);
	auto iy = get_y_index(r.orig.y, ymax, ymin, ny);

	auto index = iy * nx + ix;

	const float ncrit = 1e-6f * (std::pow(omega, 2.0f) * m_e * e_0 / std::pow(e_c, 2.0f));
	const float dt = courant_mult * std::min(dx, dy) / c;

	const float wpe = std::sqrt(eg.eden[index] * 1.0e6f * std::pow(e_c, 2.0f) / (m_e * e_0));
	const float k = std::sqrt((std::pow(omega, 2.0f) - std::pow(wpe, 2.0f)) / std::pow(c, 2.0f));

	Vec k1 = k * unit_vector(r.dir);
	Vec v1 = (k1 * std::pow(c, 2.0f)) / omega;

	r.append_path(r.orig, v1);    // Initial position

	for (int t = 1; t < nt; t++) {      // Starts from 1, origin is already in path
		Vec cur_v = r.group_v[t - 1] - std::pow(c, 2.0f) / (2.0f * ncrit) * eg.d_eden[index] * dt;
		Point cur_p = r.path[t - 1] + cur_v * dt;

		if (ray_out_of_range(cur_p)) {
			break;
		}

		r.append_path(cur_p, cur_v);

		// Current cell coordinates
		int x_pos = get_x_index(cur_p.x, xmax, xmin, nx);
		int y_pos = get_y_index(cur_p.y, ymax, ymin, ny);

		// Track cells that ray passes through
		if (x_pos != ix || y_pos != iy) {
			ix = x_pos;
			iy = y_pos;
			index = iy * nx + ix;
		}

		float xp = (cur_p.x - (get_x_val(x_pos, xmax, xmin, nx) - (dx / 2.0f))) / dx;
		float yp = (cur_p.y - (get_y_val(y_pos, ymax, ymin, ny) - (dy / 2.0f))) / dy;

		float dl = std::abs(xp);
		float dm = std::abs(yp);

		// If uray is never updated, increment = init_power
		float a1 = (1.0f - dl) * (1.0f - dm) * r.power;
		float a2 = (1.0f - dl) * dm          * r.power;
		float a3 = dl * (1.0f - dm)          * r.power;
		float a4 = dl * dm                   * r.power;

		if (xp >= 0 && yp >= 0) {
			auto a1_ind = (y_pos - 1) * nx + (x_pos + 1);
			auto a2_ind = (y_pos - 2) * nx + (x_pos + 1);
			auto a3_ind = (y_pos - 1) * nx + (x_pos + 2);
			auto a4_ind = (y_pos - 2) * nx + (x_pos + 2);

			edep[a1_ind] += a1;
			edep[a2_ind] += a2;
			edep[a3_ind] += a3;
			edep[a4_ind] += a4;
		}
		else if (xp < 0 && yp >= 0) {
			auto a1_ind = (y_pos - 1) * nx + (x_pos + 1);
			auto a2_ind = (y_pos - 2) * nx + (x_pos + 1);
			auto a3_ind = (y_pos - 1) * nx + (x_pos + 0);
			auto a4_ind = (y_pos - 2) * nx + (x_pos + 0);

			edep[a1_ind] += a1;
			edep[a2_ind] += a2;
			edep[a3_ind] += a3;
			edep[a4_ind] += a4;
		}
		else if (xp >= 0 && yp < 0) {
			auto a1_ind = (y_pos - 1) * nx + (x_pos + 1);
			auto a2_ind = (y_pos - 0) * nx + (x_pos + 1);
			auto a3_ind = (y_pos - 1) * nx + (x_pos + 2);
			auto a4_ind = (y_pos - 0) * nx + (x_pos + 2);

			edep[a1_ind] += a1;
			edep[a2_ind] += a2;
			edep[a3_ind] += a3;
			edep[a4_ind] += a4;
		}
		else if (xp < 0 && yp < 0) {
			auto a1_ind = (y_pos - 1) * nx + (x_pos + 1);
			auto a2_ind = (y_pos - 0) * nx + (x_pos + 1);
			auto a3_ind = (y_pos - 1) * nx + (x_pos + 0);
			auto a4_ind = (y_pos - 0) * nx + (x_pos + 0);

			edep[a1_ind] += a1;
			edep[a2_ind] += a2;
			edep[a3_ind] += a3;
			edep[a4_ind] += a4;
		}
		else {
			printf("Error in deposition grid interpolation.\n");
			assert(0);
		}
	}
}

#endif //CUCBET_RAY_CUH