#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include "Ray.cuh"
#include "Egrid.cuh"

class Beam {
public:
	~Beam() {
		checkErr(cudaDeviceSynchronize())
		checkErr(cudaFree(rays))
		checkErr(cudaFree(edep))
		checkErr(cudaFree(edep_new))
		checkErr(cudaFree(present))
	}

	void allocate(int beam_id, const vec2& ndir) {
		id = beam_id;
		dir = ndir;

		int big = (nx + 2) * (ny + 2);
		int lit = nx * ny;

		checkErr(cudaMallocManaged(&rays,     nrays * sizeof(Ray)))
		checkErr(cudaMallocManaged(&edep,     big   * sizeof(float)))
		checkErr(cudaMallocManaged(&edep_new, big   * sizeof(float)))
		checkErr(cudaMallocManaged(&present,  lit   * sizeof(float)))
		checkErr(cudaDeviceSynchronize())
	}
	
public:
	int id;
	Vec dir;

	Ray* rays{};
	float* edep{};  // nx + 2
	float* edep_new{}; // nx + 2
	float* present{};  // nx
};

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