#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"
#include "Egrid.cuh"

class Ray: public Managed {
public:
	Ray(const Point& origin, const Vec& direction, float power):
		orig(origin),
		init_dir(direction),
		init_power(power)
	{}

	void allocate(param_struct *pm) {
		checkErr(cudaMallocManaged(&intersections, pm->nt * sizeof(int)));
		checkErr(cudaMallocManaged(&group_v,       pm->nt * sizeof(Vec)));
		checkErr(cudaMallocManaged(&path,          pm->nt * sizeof(Point)));
		checkErr(cudaDeviceSynchronize());
	}

	void deallocate() const {
		checkErr(cudaDeviceSynchronize());
		checkErr(cudaFree(intersections));
		checkErr(cudaFree(group_v));
		checkErr(cudaFree(path));
	}

public:
	Point orig;
	Vec init_dir;
	float init_power{};

	int* intersections{};
	Vec* group_v{};
	Point* path{};
};

bool ray_out_of_range(Point* p, param_struct* pm) {
	bool valid_x = (p->x() < (pm->xmin - (pm->dx / 2.0f)) || p->x() > (pm->xmax + (pm->dx / 2.0f)));
	bool valid_z = (p->y() < (pm->ymin - (pm->dy / 2.0f)) || p->y() > (pm->ymax + (pm->dy / 2.0f)));
	return (valid_x || valid_z);
}

void draw_init_path(Ray* r, param_struct* pm, Egrid* eg, float* edep, float* present) {
	auto ix = get_x_index(r->orig.x(), pm->xmax, pm->xmin, pm->nx);
	auto iy = get_y_index(r->orig.y(), pm->ymax, pm->ymin, pm->ny);

	auto index = iy * pm->nx + ix;

	auto track_x = ix;
	auto track_y = iy;
	present[index] += 1.0f;

	float wpe = std::sqrt(eg->eden[index] * 1.0e6f * std::pow(pm->e_c, 2.0f) / (pm->m_e * pm->e_0));
	float k = std::sqrt((std::pow(pm->omega, 2.0f) - std::pow(wpe, 2.0f)) / std::pow(pm->c, 2.0f));

	Vec k1 = k * unit_vector(r->init_dir);
	Vec v1 = (k1 * std::pow(pm->c, 2.0f)) / pm->omega;

	r->path[0] = r->orig;    // Initial position
	r->group_v[0] = v1;     // Initial group velocity

	for (int t = 1; t < pm->nt; t++) {      // Starts from 1, origin is already in path
		Vec cur_v = r->group_v[t - 1] - std::pow(pm->c, 2.0f) / (2.0f * pm->ncrit) * eg->d_eden[index] * pm->dt;
		Point cur_p = r->path[t - 1] + cur_v * pm->dt;

		if (ray_out_of_range(&cur_p, pm)) {
			break;
		}

		r->path[t] = cur_p;
		r->group_v[t] = cur_v;

		int x_pos = get_x_index(cur_p.x(), pm->xmax, pm->xmin, pm->nx);
		int y_pos = get_y_index(cur_p.y(), pm->ymax, pm->ymin, pm->ny);

		// Track cells that ray passes through
		if (x_pos != track_x || y_pos != track_y) {
			auto track_index = track_y * pm->nx + track_x;
			present[track_index] += 1.0f;
			track_x = x_pos;
			track_y = y_pos;
		}

		float xp = (cur_p.x() - (get_x_val(x_pos, pm->xmax, pm->xmin, pm->nx) - (pm->dx / 2.0f))) / pm->dx;
		float yp = (cur_p.y() - (get_y_val(y_pos, pm->ymax, pm->ymin, pm->ny) - (pm->dy / 2.0f))) / pm->dy;

		float dl = std::abs(xp);
		float dm = std::abs(yp);

		// If uray is never updated, increment = init_power
		float a1 = (1.0f - dl) * (1.0f - dm) * r->init_power;
		float a2 = (1.0f - dl) * dm          * r->init_power;
		float a3 = dl * (1.0f - dm)          * r->init_power;
		float a4 = dl * dm                   * r->init_power;

		if (xp >= 0 && yp >= 0) {
			auto a1_ind = (y_pos - 1) * pm->nx + (x_pos + 1);
			auto a2_ind = (y_pos - 2) * pm->nx + (x_pos + 1);
			auto a3_ind = (y_pos - 1) * pm->nx + (x_pos + 2);
			auto a4_ind = (y_pos - 2) * pm->nx + (x_pos + 2);

			edep[a1_ind] += a1;
			edep[a2_ind] += a2;
			edep[a3_ind] += a3;
			edep[a4_ind] += a4;
		}
		else if (xp < 0 && yp >= 0) {
			auto a1_ind = (y_pos - 1) * pm->nx + (x_pos + 1);
			auto a2_ind = (y_pos - 2) * pm->nx + (x_pos + 1);
			auto a3_ind = (y_pos - 1) * pm->nx + (x_pos + 0);
			auto a4_ind = (y_pos - 2) * pm->nx + (x_pos + 0);

			edep[a1_ind] += a1;
			edep[a2_ind] += a2;
			edep[a3_ind] += a3;
			edep[a4_ind] += a4;
		}
		else if (xp >= 0 && yp < 0) {
			auto a1_ind = (y_pos - 1) * pm->nx + (x_pos + 1);
			auto a2_ind = (y_pos - 0) * pm->nx + (x_pos + 1);
			auto a3_ind = (y_pos - 1) * pm->nx + (x_pos + 2);
			auto a4_ind = (y_pos - 0) * pm->nx + (x_pos + 2);

			edep[a1_ind] += a1;
			edep[a2_ind] += a2;
			edep[a3_ind] += a3;
			edep[a4_ind] += a4;
		}
		else if (xp < 0 && yp < 0) {
			auto a1_ind = (y_pos - 1) * pm->nx + (x_pos + 1);
			auto a2_ind = (y_pos - 0) * pm->nx + (x_pos + 1);
			auto a3_ind = (y_pos - 1) * pm->nx + (x_pos + 0);
			auto a4_ind = (y_pos - 0) * pm->nx + (x_pos + 0);

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