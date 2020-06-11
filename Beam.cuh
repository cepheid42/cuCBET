#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include "Ray.cuh"

class Beam: public Managed {
public:
	Beam(int beam_id, const vec2& dir) : id(beam_id), dir(dir) {
		int big = (nx + 2) * (ny + 2);
		int little = nx * ny;

		checkErr(cudaMallocManaged(&rays,     nrays  * sizeof(Ray)));
		for(int r = 0; r < nrays; r++) {
			rays[r] = new Ray; // init with blanks
		}
		checkErr(cudaMallocManaged(&edep,     big    * sizeof(float)));
		checkErr(cudaMallocManaged(&edep_new, big    * sizeof(float)));
		checkErr(cudaMallocManaged(&present,  little * sizeof(float)));
		checkErr(cudaDeviceSynchronize());
	}

	~Beam() {
		checkErr(cudaDeviceSynchronize());
		checkErr(cudaMallocManaged(&rays,     nrays  * sizeof(Ray)));
		checkErr(cudaFree(rays));
		checkErr(cudaFree(edep));
		checkErr(cudaFree(edep_new));
		checkErr(cudaFree(present));
	}
	
public:
	int id;
	Vec dir;

	Ray* rays;
	float* edep;  // nx + 2
	float* edep_new; // nx + 2
	float* present;  // nx
};

//__device__ inline float determinate(Point& p1, Point& p2, Point& p3, Point& p4) {
//	auto numerator = ((p1.x() - p3.x()) * (p3.y() - p4.y()) - (p1.y() - p3.y()) * (p3.x() - p4.x()));
//	auto denominator = ((p1.x() - p2.x()) * (p3.y() - p4.y()) - (p1.y() - p2.y()) * (p3.x() - p4.x()));
//
//	if (denominator == 0) {
//		// lines are parallel
//		return -1.0f;
//	}
//
//	return (numerator / denominator);
//
//}
//
//__global__ void find_intersections(Ray& r1, Ray& r2, param_struct& pm) {
//	int tx = threadIdx.x + blockIdx.x * blockDim.x;
//	int stride = blockDim.x * gridDim.x;
//
//	for (int t1 = tx; t1 < nt; t1 += stride) {
//		// Prevents out of bounds accesses?
//		if (t1 + 1 >= nt) {
//			break;
//		}
//
//		auto p1 = r1.path[t1];
//		auto p2 = r1.path[t1 + 1];
//
//		for (int t2 = 0; t2 < nt - 1; t2++) {
//			auto p3 = r2.path[t2];
//			auto p4 = r2.path[t2 + 1];
//
//			auto t = determinate(p1, p2, p3, p4);
//
//			if (t == -1.0f){
//				// lines are parallel... somehow
//				break;
//			}
//
//			if (0.0f <= t && t <= 1.0f) {
//				// Found intersection
//				r1.intersections[t1] = t1;
//				r2.intersections[t2] = t2;
//			}
//		}
//	}
//}
//
//void get_intersections(Beam& b1, Beam& b2, param_struct& pm) {
//	int blockSize = 1024;
//	int numBlocks = (nrays + blockSize - 1) / blockSize;
//
//	for (int r = 0; r < nrays; r++) {
//		Ray ray1 = b1.rays[r];
//
//		for (int p = 0; p < nrays; p++) {
//			Ray ray2 = b2.rays[p];
//
//			find_intersections<<<numBlocks, blockSize>>>(ray1, ray2, pm);
//			checkErr(cudaDeviceSynchronize());
//		}
//	}
//}
//
//void cpu_find_intersections(Beam& b1, Beam& b2, param_struct& pm) {
//	for (int r1 = 0; r1 < nrays; r1++) {
//		Ray ray1 = b1.rays[r1];
//		for (int t1 = 0; t1 < nt; t1++) {
//			auto p1 = ray1.path[t1];
//			auto ix1 = get_x_index(p1.x(), xmax, xmin, nx);
//			auto iy1 = get_y_index(p1.y(), ymax, ymin, ny);
//
//			float min_dist = xmax - xmin;
//
//			for (int r2 = 0; r2 < nrays; r2++) {
//				Ray ray2 = b2.rays[r2];
//
//				for (int t2 = 0; t2 < nt; t2++) {
//					auto p2 = ray2.path[t2];
//					auto ix2 = get_x_index(p2.x(), xmax, xmin, nx);
//					auto iy2 = get_y_index(p2.y(), ymax, ymin, ny);
//					int min_index = -1;
//
//					if (ix1 == ix2 && iy1 == iy2) {
//						auto dist = (p2 - p1).length();
//						if (dist <= min_dist) {
//							min_dist = dist;
//							min_index = t2;
//						}
//					}
//					if (iy1 > iy2) {   // y index increases down, so if y1 > y2 then ray2 has gone past ray1
//						if (min_index != -1) {
//							ray1.intersections[t1] = t1;
//							ray2.intersections[min_index] = min_index;
//						}
//
//						break;
//					}
//				}
//			}
//		}
//	}
//}
//
////void calc_gain(Beam& b1, Beam& b2, Egrid& eg, param_struct& pm) {
////	for (auto& r1: b1.rays) {
////		for (auto i1: r1.intersections) {
////			auto p1 = r1.path[i1];
////			auto ix1 = get_x_index(p1.x(), xmax, xmin, nx);
////			auto iy1 = get_y_index(p1.y(), ymax, ymin, ny);
////
////			for (auto& r2: b2.rays) {
////				for (auto i2: r2.intersections){
////					auto p2 = r2.path[i2];
////					auto ix2 = get_x_index(p2.x(), xmax, xmin, nx);
////					auto iy2 = get_y_index(p2.y(), ymax, ymin, ny);
////
////					if (ix1 == ix2 && iy1 == iy2) { // Double check, why not.
////						auto next_p1 = r1.path[i1 + 1];
////						auto dk1 = next_p1 - p1;
////
////						auto next_p2 = r2.path[i2 + 1];
////						auto dk2 = next_p2 - p2;
////
////						auto ne = eg.eden[iy1][ix1];
////						auto epsilon = std::sqrt(1.0f - ne / ncrit);
////
////						auto kmag = (omega / c) * epsilon;
////
////						auto k1 = kmag * unit_vector(dk1);
////						auto k2 = kmag * unit_vector(dk2);
////
////						auto kiaw = std::sqrt(std::pow(k2.x() - k1.x(), 2.0f) + std::pow(k2.y() - k1.y(), 2.0f));
////						auto ws = kiaw * cs;
////
////						auto machnum = (((-0.4f) - (-2.4f)) / (xmax - xmin)) * (get_x_val(ix1, xmax, xmin, nx) - xmin) + (-2.4f);
////						auto u_flow = std::max(machnum, 0.0f) * cs;
////						auto eta = ((omega - omega) - (k2.x() - k1.x()) * u_flow) / (ws + 1.0e-10f);   // omega is not changed in this code
////
////						auto efield1 = std::sqrt(8.0f * pi * 1.0e7f * b1.edep[iy1][ix1] / c);
////						auto P = (std::pow(iaw, 2.0f) * eta) / (std::pow(std::pow(eta, 2.0f) - 1.0f, 2.0f) + std::pow(iaw, 2.0f) * std::pow(eta, 2.0f));
////						auto gain2 = constant1 * std::pow(efield1, 2.0f) * (ne / ncrit) * (1.0f / iaw) * P;
////
////						// Update W1_new
////						eg.W_new[iy1][ix1][0] = eg.W[iy1][ix1][0] * std::exp(1 * eg.W[iy1][ix1][1] * dk1.length() * gain2 / epsilon);
////						// Update W2_new
////						eg.W_new[iy1][ix1][1] = eg.W[iy1][ix1][1] * std::exp(-1 * eg.W[iy1][ix1][0] * dk2.length() * gain2 / epsilon);
////					}
////				}
////			}
////		}
////	}
////}
////
////void calc_intensity(Beam& b1, Beam& b2, Egrid& eg, param_struct& pm) {
////	for (auto& r1: b1.rays) {
////		for (auto i1: r1.intersections) {
////			auto p1 = r1.path[i1];
////			auto ix1 = get_x_index(p1.x(), xmax, xmin, nx);
////			auto iy1 = get_y_index(p1.y(), ymax, ymin, ny);
////
////			for (auto& r2: b2.rays) {
////				for (auto i2: r2.intersections){
////					auto p2 = r2.path[i2];
////					auto ix2 = get_x_index(p2.x(), xmax, xmin, nx);
////					auto iy2 = get_y_index(p2.y(), ymax, ymin, ny);
////
////					if (ix1 == ix2 && iy1 == iy2) { // Double check again.
////						float frac_change_1 = -1.0f * (1.0f - (eg.W_new[iy1][ix1][0] / eg.W[iy1][ix1][0])) * b1.edep[iy1][ix1];
////						float frac_change_2 = -1.0f * (1.0f - (eg.W_new[iy1][ix1][1] / eg.W[iy1][ix1][1])) * b2.edep[iy1][ix1];
////
////						b1.edep_new[iy1][ix1] += frac_change_1;
////						b2.edep_new[iy1][ix1] += frac_change_2;
////
////						for (int q1 = i1 + 1; q1 < r1.path.size(); q1++) {
////							auto ix_cur = get_x_index(r1.path[q1].x(), xmax, xmin, nx);
////							auto iy_cur = get_y_index(r1.path[q1].y(), ymax, ymin, ny);
////
////							if (ix_cur == ix1 || iy_cur == iy1) {   // Prevent double deposition in same zone
////								b1.edep_new[iy_cur][ix_cur] += frac_change_1 * (b1.present[iy1][ix1] / b1.present[iy_cur][ix_cur]);
////							}
////							ix1 = ix_cur;
////							iy1 = iy_cur;
////						}
////
////						for (int q2 = i2 + 1; q2 < r2.path.size(); q2++) {
////							auto ix_cur = get_x_index(r2.path[q2].x(), xmax, xmin, nx);
////							auto iy_cur = get_y_index(r2.path[q2].y(), ymax, ymin, ny);
////
////							if (ix_cur == ix2 || iy_cur == iy2) { // Prevent double deposition in same zone
////								b2.edep_new[iy_cur][ix_cur] += frac_change_1 * (b1.present[iy1][ix1] / b1.present[iy_cur][ix_cur]);
////							}
////							ix2 = ix_cur;
////							iy2 = iy_cur;
////						}
////					}
////				}
////			}
////		}
////	}
////}




#endif //CUCBET_BEAM_CUH