#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include "Ray.cuh"

class Beam: public Managed {
public:
	Beam(int beam_id, int num_rays, const vec2& dir) :
		beam_num(beam_id),
		nrays(num_rays),
		direction(dir)
	{}

	void allocate(param_struct *pm) {
		checkErr(cudaMallocManaged(&rays,      pm->nrays * sizeof(Ray)));
		checkErr(cudaMallocManaged(&edep,     (pm->nx + 2) * (pm->ny + 2) * sizeof(float)));
		checkErr(cudaMallocManaged(&edep_new, (pm->nx + 2) * (pm->ny + 2) * sizeof(Point)));
		checkErr(cudaMallocManaged(&present,   pm->nx      *  pm->ny      * sizeof(Point)));
		checkErr(cudaDeviceSynchronize());
	}

	void deallocate() const {
		checkErr(cudaDeviceSynchronize());
		for (int r = 0; r < nrays; r++) {
			rays[r]->deallocate();
		}
		checkErr(cudaFree(rays));
		checkErr(cudaFree(edep));
		checkErr(cudaFree(edep_new));
		checkErr(cudaFree(present));
	}

public:
	int beam_num;
	int nrays;
	Vec direction;
	Ray** rays{};
	float* edep{};  // nx + 2
	float* edep_new{}; // nx + 2
	float* present{};  // nx
};

void init_beam(Beam* b, Egrid* eg, param_struct* pm, float x_start, float y_start, float step) {
	// Iterate over rays
	for (int r = 0; r < b->nrays; r++) {
		float uray0;
		if (b->beam_num == 0) {
			uray0 = pm->uray_mult * pm->phase_interp.findValue(y_start);
		} else {
			uray0 = pm->uray_mult * pm->phase_interp.findValue(x_start);
		}

		Point ray_orig(x_start, y_start);

		Ray *ray1 = new Ray(ray_orig, b->direction, uray0);
		ray1->allocate(pm);

		draw_init_path(ray1, pm, eg, b->edep, b->present);

		b->rays[r] = ray1;

		if (b->beam_num == 0) {
			y_start += step;
		} else {
			x_start += step;
		}
	}
}

__global__ void find_intersections(Beam* b1, Beam* b2, param_struct* pm) {
	for (int r = 0; r < pm->nrays; r++) {
		Ray *r1 = b1->rays[r];

		for (int t1 = 0; t1 < pm->nt; t1++) {
			auto p1 = r1->path[t1];
			auto ix1 = get_x_index(p1.x(), pm->xmax, pm->xmin, pm->nx);
			auto iy1 = get_y_index(p1.y(), pm->ymax, pm->ymin, pm->ny);

			for (int p = 0; p < pm->nrays; p++) {
				Ray *r2 = b2->rays[p];

				Tuple* valid_points;
				int valid_index = 0;

				for (int t2 = 0; t2 < pm->nt; t2++) {
					auto p2 = r2->path[t2];
					auto ix2 = get_x_index(p2.x(), pm->xmax, pm->xmin, pm->nx);
					auto iy2 = get_y_index(p2.y(), pm->ymax, pm->ymin, pm->ny);

					if (ix1 == ix2 && iy1 == iy2) {
						auto dist = (p2 - p1).length();
						valid_points[valid_index] = Tuple(dist, t2);
						valid_index += 1;
						continue;
					}
					else if (iy1 > iy2) {   // y index increases down, so if y1 > y2 then ray2 has gone past ray1

						//
						//
						//
						// Fix valid_index shit
						// Linear interpolation would probably be a better choice than
						// trying to track a bunch of points of unknown number;
						//
						//
						//



						if (!valid_points.empty()) {
							auto valid_p2 = *std::min_element(valid_points.begin(), valid_points.end());
							r1->intersections.emplace_back(t);
							r2.intersections.emplace_back(valid_p2.second);
						}
						break;
					}
					else {
						continue;
					}
				}
			}
		}
	}
}

//void calc_gain(Beam& b1, Beam& b2, Egrid& eg, param_struct& pm) {
//	for (auto& r1: b1.rays) {
//		for (auto i1: r1.intersections) {
//			auto p1 = r1.path[i1];
//			auto ix1 = get_x_index(p1.x(), pm.xmax, pm.xmin, pm.nx);
//			auto iy1 = get_y_index(p1.y(), pm.ymax, pm.ymin, pm.ny);
//
//			for (auto& r2: b2.rays) {
//				for (auto i2: r2.intersections){
//					auto p2 = r2.path[i2];
//					auto ix2 = get_x_index(p2.x(), pm.xmax, pm.xmin, pm.nx);
//					auto iy2 = get_y_index(p2.y(), pm.ymax, pm.ymin, pm.ny);
//
//					if (ix1 == ix2 && iy1 == iy2) { // Double check, why not.
//						auto next_p1 = r1.path[i1 + 1];
//						auto dk1 = next_p1 - p1;
//
//						auto next_p2 = r2.path[i2 + 1];
//						auto dk2 = next_p2 - p2;
//
//						auto ne = eg.eden[iy1][ix1];
//						auto epsilon = std::sqrt(1.0f - ne / pm.ncrit);
//
//						auto kmag = (pm.omega / pm.c) * epsilon;
//
//						auto k1 = kmag * unit_vector(dk1);
//						auto k2 = kmag * unit_vector(dk2);
//
//						auto kiaw = std::sqrt(std::pow(k2.x() - k1.x(), 2.0f) + std::pow(k2.y() - k1.y(), 2.0f));
//						auto ws = kiaw * pm.cs;
//
//						auto machnum = (((-0.4f) - (-2.4f)) / (pm.xmax - pm.xmin)) * (get_x_val(ix1, pm.xmax, pm.xmin, pm.nx) - pm.xmin) + (-2.4f);
//						auto u_flow = std::max(machnum, 0.0f) * pm.cs;
//						auto eta = ((pm.omega - pm.omega) - (k2.x() - k1.x()) * u_flow) / (ws + 1.0e-10f);   // omega is not changed in this code
//
//						auto efield1 = std::sqrt(8.0f * pm.pi * 1.0e7f * b1.edep[iy1][ix1] / pm.c);
//						auto P = (std::pow(pm.iaw, 2.0f) * eta) / (std::pow(std::pow(eta, 2.0f) - 1.0f, 2.0f) + std::pow(pm.iaw, 2.0f) * std::pow(eta, 2.0f));
//						auto gain2 = pm.constant1 * std::pow(efield1, 2.0f) * (ne / pm.ncrit) * (1.0f / pm.iaw) * P;
//
//						// Update W1_new
//						eg.W_new[iy1][ix1][0] = eg.W[iy1][ix1][0] * std::exp(1 * eg.W[iy1][ix1][1] * dk1.length() * gain2 / epsilon);
//						// Update W2_new
//						eg.W_new[iy1][ix1][1] = eg.W[iy1][ix1][1] * std::exp(-1 * eg.W[iy1][ix1][0] * dk2.length() * gain2 / epsilon);
//					}
//				}
//			}
//		}
//	}
//}
//
//void calc_intensity(Beam& b1, Beam& b2, Egrid& eg, param_struct& pm) {
//	for (auto& r1: b1.rays) {
//		for (auto i1: r1.intersections) {
//			auto p1 = r1.path[i1];
//			auto ix1 = get_x_index(p1.x(), pm.xmax, pm.xmin, pm.nx);
//			auto iy1 = get_y_index(p1.y(), pm.ymax, pm.ymin, pm.ny);
//
//			for (auto& r2: b2.rays) {
//				for (auto i2: r2.intersections){
//					auto p2 = r2.path[i2];
//					auto ix2 = get_x_index(p2.x(), pm.xmax, pm.xmin, pm.nx);
//					auto iy2 = get_y_index(p2.y(), pm.ymax, pm.ymin, pm.ny);
//
//					if (ix1 == ix2 && iy1 == iy2) { // Double check again.
//						float frac_change_1 = -1.0f * (1.0f - (eg.W_new[iy1][ix1][0] / eg.W[iy1][ix1][0])) * b1.edep[iy1][ix1];
//						float frac_change_2 = -1.0f * (1.0f - (eg.W_new[iy1][ix1][1] / eg.W[iy1][ix1][1])) * b2.edep[iy1][ix1];
//
//						b1.edep_new[iy1][ix1] += frac_change_1;
//						b2.edep_new[iy1][ix1] += frac_change_2;
//
//						for (int q1 = i1 + 1; q1 < r1.path.size(); q1++) {
//							auto ix_cur = get_x_index(r1.path[q1].x(), pm.xmax, pm.xmin, pm.nx);
//							auto iy_cur = get_y_index(r1.path[q1].y(), pm.ymax, pm.ymin, pm.ny);
//
//							if (ix_cur == ix1 || iy_cur == iy1) {   // Prevent double deposition in same zone
//								b1.edep_new[iy_cur][ix_cur] += frac_change_1 * (b1.present[iy1][ix1] / b1.present[iy_cur][ix_cur]);
//							}
//							ix1 = ix_cur;
//							iy1 = iy_cur;
//						}
//
//						for (int q2 = i2 + 1; q2 < r2.path.size(); q2++) {
//							auto ix_cur = get_x_index(r2.path[q2].x(), pm.xmax, pm.xmin, pm.nx);
//							auto iy_cur = get_y_index(r2.path[q2].y(), pm.ymax, pm.ymin, pm.ny);
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

// Utility Functions
void save_beam_to_file(Beam* beam, const std::string& beam_name, param_struct* pm) {
	const std::string output_path = "./Outputs/" + beam_name;

	// Write beam to file
	std::ofstream beam_file(output_path + ".csv");
	beam_file << std::setprecision(std::numeric_limits<float>::max_digits10);

	for (int r = 0; r < pm->nrays; r++) {
		Ray* cur_ray = beam->rays[r];
		for (int t = 0; t < pm->nt; t++) {
			beam_file << r << ", " << cur_ray->path[t] << "\n";
		}
	}
	beam_file.close();

	// Write beam edep to file (i_b#)
	std::ofstream edep_file(output_path + "_edep.csv");
	edep_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int y = pm->ny - 1; y >= 0; y--) {
		for (int x = 0; x < pm->nx; x++) {
			int index = y * pm->nx + x;
			edep_file << beam->edep[index];
			if (x != pm->nx - 1) {
				edep_file << ", ";
			}
		}
		edep_file << "\n";
	}
	edep_file.close();

	// Write beam edep_new to file (i_b#_new)
	std::ofstream edep_new_file(output_path + "_edep_new.csv");
	edep_new_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int y = pm->ny - 1; y >= 0; y--) {
		for (int x = 0; x < pm->nx; x++) {
			int index = y * pm->nx + x;
			edep_new_file << beam->edep[index];
			if (x != pm->nx - 1) {
				edep_new_file << ", ";
			}
		}
		edep_new_file << "\n";
	}
	edep_new_file.close();
}

#endif //CUCBET_BEAM_CUH