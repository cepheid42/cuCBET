#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include "Ray.cuh"
#include "Interpolator.cuh"

class Beam {
public:
	Beam() : rays(nullptr), edep(nullptr), edep_new(nullptr), id(-1), dir() {}
	Beam(Beam&) = delete;
	~Beam() = default;

	Beam& operator=(Beam&) = delete;

	void allocate(const int beam_id, const float x_dir, const float y_dir) {
		id = beam_id;
		dir.x = x_dir;
		dir.y = y_dir;

		checkErr(cudaMallocManaged(&rays, nrays * sizeof(Ray)))
		checkErr(cudaMallocManaged(&edep, (nx + 2) * (ny + 2) * sizeof(float)))
		checkErr(cudaMallocManaged(&edep_new, (nx + 2) * (ny + 2) * sizeof(float)))
		checkErr(cudaDeviceSynchronize())
	}

	void free_mem() const {
		checkErr(cudaDeviceSynchronize())
		for(int i = 0; i < nrays; i++) {
			rays[i].free_mem();
		}
		checkErr(cudaFree(rays))
		checkErr(cudaFree(edep))
		checkErr(cudaFree(edep_new))
	}
	
public:
	Ray* rays;
	float* edep;  // nx + 2
	float* edep_new; // nx + 2
	int id;
	Vec dir;
};

void init_beam(Beam& b, Egrid& eg, float x_start, float y_start, float step) {
	Interpolator interp = new_interpolator();
	const int nt = int(2.0f * std::max(nx, ny) / courant_mult);

	// Iterate over rays
	for (int r = 0; r < nrays; r++) {
		float uray0;
		if (b.id == 0) {
			uray0 = uray_mult * interp.findValue(y_start);
		}
		if (b.id == 1) {
			uray0 = uray_mult * interp.findValue(x_start);
		}

		b.rays[r].allocate(x_start, y_start, b.dir, uray0, nt);
		draw_init_path(b.rays[r], eg, b.edep, nt);

		if (b.id == 0) {
			y_start += step;
		}
		if (b.id == 1){
			x_start += step;
		}
	}
}

//void calc_intensity(Beam& b1, Beam& b2, Egrid& eg, param_struct& pm) {
//	for (auto& r1: b1.rays) {
//		for (auto i1: r1.intersections) {
//			auto p1 = r1.path[i1];
//			auto ix1 = get_x_index(p1.x(), xmax, xmin, nx);
//			auto iy1 = get_y_index(p1.y(), ymax, ymin, ny);
//
//			for (auto& r2: b2.rays) {
//				for (auto i2: r2.intersections){
//					auto p2 = r2.path[i2];
//					auto ix2 = get_x_index(p2.x(), xmax, xmin, nx);
//					auto iy2 = get_y_index(p2.y(), ymax, ymin, ny);
//
//					if (ix1 == ix2 && iy1 == iy2) { // Double check again.
//						float frac_change_1 = -1.0f * (1.0f - (eg.W_new[iy1][ix1][0] / eg.W[iy1][ix1][0])) * b1.edep[iy1][ix1];
//						float frac_change_2 = -1.0f * (1.0f - (eg.W_new[iy1][ix1][1] / eg.W[iy1][ix1][1])) * b2.edep[iy1][ix1];
//
//						b1.edep_new[iy1][ix1] += frac_change_1;
//						b2.edep_new[iy1][ix1] += frac_change_2;
//
//						for (int q1 = i1 + 1; q1 < r1.path.size(); q1++) {
//							auto ix_cur = get_x_index(r1.path[q1].x(), xmax, xmin, nx);
//							auto iy_cur = get_y_index(r1.path[q1].y(), ymax, ymin, ny);
//
//							if (ix_cur == ix1 || iy_cur == iy1) {   // Prevent double deposition in same zone
//								b1.edep_new[iy_cur][ix_cur] += frac_change_1 * (b1.present[iy1][ix1] / b1.present[iy_cur][ix_cur]);
//							}
//							ix1 = ix_cur;
//							iy1 = iy_cur;
//						}
//
//						for (int q2 = i2 + 1; q2 < r2.path.size(); q2++) {
//							auto ix_cur = get_x_index(r2.path[q2].x(), xmax, xmin, nx);
//							auto iy_cur = get_y_index(r2.path[q2].y(), ymax, ymin, ny);
//
//							if (ix_cur == ix2 || iy_cur == iy2) { // Prevent double deposition in same zone
//								b2.edep_new[iy_cur][ix_cur] += frac_change_1 * (b1.present[iy1][ix1] / b1.present[iy_cur][ix_cur]);
//							}
//							ix2 = ix_cur;
//							iy2 = iy_cur;
//						}
//					}
//				}
//			}
//		}
//	}
//}

#endif //CUCBET_BEAM_CUH