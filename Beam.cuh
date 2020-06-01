#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include "Ray.cuh"
#include "Interpolator.cuh"
#include <iostream>

class Beam {
public:
	Beam() = delete;
	Beam(int beam_id, int num_rays, const Vec& dir) :
		beam_num(beam_id),
		nrays(num_rays)
	{
		direction = dir;
		rays.reserve(nrays); // Pre-size ray vector
	}
public:
	int beam_num;
	int nrays;
	Vec direction;
	std::vector<Ray> rays;
	std::array<std::array<float, nx + 2>, ny + 2> edep{};  // Each beam tracks it's electron deposition
	std::array<std::array<float, nx + 2>, ny + 2> edep_new{};
	array<array<float, nx>, ny> present{};
};

void init_beam(Beam& b, Egrid& eg, float x_start, float y_start, float step, float dt, int nt, float ncrit, Interpolator& interp) {
	// Iterate over rays
	for (int r = 0; r < b.nrays; r++) {
		float uray0;
		if (b.beam_num == 0) {
			// First beam goes left to right
			uray0 = uray_mult * interp.findValue(y_start + offset);
		} else {
			// Second beam goes bottom to top
			uray0 = uray_mult * interp.findValue(x_start);
		}

		Point ray_orig(x_start, y_start);
		Ray ray1(ray_orig, b.direction, uray0, nt);
		draw_init_path(ray1, nt, dt, ncrit, eg, b.edep, b.present);
		b.rays.emplace_back(ray1);

		if (b.beam_num == 0) {
			y_start += step;
		} else {
			x_start += step;
		}
	}
}

void find_intersections(Beam& b1, Beam& b2, float dt) {
	for (auto& r1: b1.rays) {
		for (int i = 0; i < r1.path.size(); i++) {
			auto p1 = r1.path[i];
			auto ix1 = get_x_index(p1.x(), xmax, xmin, nx);
			auto iy1 = get_y_index(p1.y(), ymax, ymin, ny);

			for (auto& r2: b2.rays) {
				std::vector<std::pair<float, int>> valid_points;

				for (int j = 0; j < r2.path.size(); j++) {
					auto p2 = r2.path[j];
					auto ix2 = get_x_index(p2.x(), xmax, xmin, nx);
					auto iy2 = get_y_index(p2.y(), ymax, ymin, ny);

					if (ix1 == ix2 && iy1 == iy2) {
						auto dist = (p2 - p1).length();
						valid_points.emplace_back(std::pair<float, int>(dist, j));
						continue;
					}
					else if (iy1 > iy2) {   // y index increases down, so if y1 > y2 then ray2 has gone past ray1
						if (!valid_points.empty()) {
							auto valid_p2 = *std::min_element(valid_points.begin(), valid_points.end());
							r1.intersections.emplace_back(i);
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

void calc_gain(Beam& b1, Beam& b2, Egrid& eg, float ncrit) {
	for (auto& r1: b1.rays) {
		for (auto i1: r1.intersections) {
			auto p1 = r1.path[i1];
			auto ix1 = get_x_index(p1.x(), xmax, xmin, nx);
			auto iy1 = get_y_index(p1.y(), ymax, ymin, ny);

			for (auto& r2: b2.rays) {
				for (auto i2: r2.intersections){
					auto p2 = r2.path[i2];
					auto ix2 = get_x_index(p2.x(), xmax, xmin, nx);
					auto iy2 = get_y_index(p2.y(), ymax, ymin, ny);

					if (ix1 == ix2 && iy1 == iy2) { // Double check, why not.
						auto next_p1 = r1.path[i1 + 1];
						auto dk1 = next_p1 - p1;

						auto next_p2 = r2.path[i2 + 1];
						auto dk2 = next_p2 - p2;

						auto ne = eg.eden[iy1][ix1];
						auto epsilon = std::sqrt(1.0f - ne / ncrit);

						auto kmag = (omega / c) * epsilon;

						auto k1 = kmag * unit_vector(dk1);
						auto k2 = kmag * unit_vector(dk2);

						auto kiaw = std::sqrt(std::pow(k2.x() - k1.x(), 2.0f) + std::pow(k2.y() - k1.y(), 2.0f));
						auto ws = kiaw * cs;

						auto machnum = (((-0.4f) - (-2.4f)) / (xmax - xmin)) * (get_x_val(ix1, xmax, xmin, nx) - xmin) + (-2.4f);
						auto u_flow = std::max(machnum, 0.0f) * cs;
						auto eta = ((omega - omega) - (k2.x() - k1.x()) * u_flow) / (ws + 1.0e-10f);   // omega is not changed in this code

						auto efield1 = std::sqrt(8.0f * pi * 1.0e7f * b1.edep[iy1][ix1] / c);
						auto P = (std::pow(iaw, 2.0f) * eta) / (std::pow(std::pow(eta, 2.0f) - 1.0f, 2.0f) + std::pow(iaw, 2.0f) * std::pow(eta, 2.0f));
						auto gain2 = constant1 * std::pow(efield1, 2.0f) * (ne / ncrit) * (1.0f / iaw) * P;

						// Update W1_new
						eg.W_new[iy1][ix1][0] = eg.W[iy1][ix1][0] * std::exp(1 * eg.W[iy1][ix1][1] * dk1.length() * gain2 / epsilon);
						// Update W2_new
						eg.W_new[iy1][ix1][1] = eg.W[iy1][ix1][1] * std::exp(-1 * eg.W[iy1][ix1][0] * dk2.length() * gain2 / epsilon);
					}
				}
			}
		}
	}
}

void calc_intensity(Beam& b1, Beam& b2, Egrid& eg) {
	for (auto& r1: b1.rays) {
		for (auto i1: r1.intersections) {
			auto p1 = r1.path[i1];
			auto ix1 = get_x_index(p1.x(), xmax, xmin, nx);
			auto iy1 = get_y_index(p1.y(), ymax, ymin, ny);

			for (auto& r2: b2.rays) {
				for (auto i2: r2.intersections){
					auto p2 = r2.path[i2];
					auto ix2 = get_x_index(p2.x(), xmax, xmin, nx);
					auto iy2 = get_y_index(p2.y(), ymax, ymin, ny);

					if (ix1 == ix2 && iy1 == iy2) { // Double check again.
						float frac_change_1 = -1.0f * (1.0f - (eg.W_new[iy1][ix1][0] / eg.W[iy1][ix1][0])) * b1.edep[iy1][ix1];
						float frac_change_2 = -1.0f * (1.0f - (eg.W_new[iy1][ix1][1] / eg.W[iy1][ix1][1])) * b2.edep[iy1][ix1];

						b1.edep_new[iy1][ix1] += frac_change_1;
						b2.edep_new[iy1][ix1] += frac_change_2;

						for (int q1 = i1 + 1; q1 < r1.path.size(); q1++) {
							auto ix_cur = get_x_index(r1.path[q1].x(), xmax, xmin, nx);
							auto iy_cur = get_y_index(r1.path[q1].y(), ymax, ymin, ny);

							if (ix_cur == ix1 || iy_cur == iy1) {   // Prevent double deposition in same zone
								b1.edep_new[iy_cur][ix_cur] += frac_change_1 * (b1.present[iy1][ix1] / b1.present[iy_cur][ix_cur]);
							}
							ix1 = ix_cur;
							iy1 = iy_cur;
						}

						for (int q2 = i2 + 1; q2 < r2.path.size(); q2++) {
							auto ix_cur = get_x_index(r2.path[q2].x(), xmax, xmin, nx);
							auto iy_cur = get_y_index(r2.path[q2].y(), ymax, ymin, ny);

							if (ix_cur == ix2 || iy_cur == iy2) { // Prevent double deposition in same zone
								b2.edep_new[iy_cur][ix_cur] += frac_change_1 * (b1.present[iy1][ix1] / b1.present[iy_cur][ix_cur]);
							}
							ix2 = ix_cur;
							iy2 = iy_cur;
						}
					}
				}
			}
		}
	}
}

// Utility Functions
void save_beam_to_file(Beam& beam, const std::string& beam_name) {
	const std::string output_path = "./Outputs/";

	// Write beam to file
	std::ofstream beam_file(output_path + beam_name + ".csv");
	beam_file << std::setprecision(std::numeric_limits<float>::max_digits10);

	for (int r = 0; r < beam.rays.size(); ++r) {
		for (int i = 0; i < beam.rays[r].path.size(); i++) {
			beam_file << r << ", " << beam.rays[r].path[i] << "\n";
		}
	}
	beam_file.close();

	// Write beam edep to file (i_b#)
	std::ofstream edep_file(output_path + beam_name + "_edep.csv");
	edep_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int i = ny - 1; i >= 0; i--) {
		for (int j = 0; j < nx; j++) {
			edep_file << beam.edep[i][j];
			if (j != nx - 1) {
				edep_file << ", ";
			}
		}
		edep_file << "\n";
	}
	edep_file.close();

	// Write beam edep_new to file (i_b#_new)
	std::ofstream edep_new_file(output_path + beam_name + "_edep_new.csv");
	edep_new_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int i = ny - 1; i >= 0; i--) {
		for (int j = 0; j < nx; j++) {
			edep_new_file << beam.edep[i][j];
			if (j != nx - 1) {
				edep_new_file << ", ";
			}
		}
		edep_new_file << "\n";
	}
	edep_new_file.close();
}

#endif //CUCBET_BEAM_CUH