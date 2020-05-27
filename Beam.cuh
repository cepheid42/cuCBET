#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include "Ray.cuh"
#include "Interpolator.cuh"


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
};

void init_beam(Beam& b, Egrid& eg, float x_start, float y_start, float step, float dt, int nt, float ncrit, Interpolator& interp) {
	// Iterate over rays
	for (int r = 0; r < b.nrays; r++) {
		float uray0;
		if (b.beam_num == 0) {
			// First beam goes left to right
			uray0 = uray_mult * interp.findValue(y_start);
		} else {
			// Second beam goes bottom to top
			uray0 = uray_mult * interp.findValue(x_start);
		}

		Point ray_orig(x_start, y_start);
		Ray ray1(ray_orig, b.direction, uray0, nt);
		draw_init_path(ray1, nt, dt, ncrit, eg, b.edep);
		b.rays.emplace_back(ray1);

		if (b.beam_num == 0) {
			y_start += step;
		} else {
			x_start += step;
		}
	}
}


void calc_gain(Beam& b1, Beam& b2, Egrid& eg, float ncrit) {
	for (auto& r1: b1.rays) {   // Rays in Beam 1
		auto p1_start = get_x_index(b2.rays.front().orig.x(), xmax, xmin, nx);
		auto p1_end = get_x_index(b2.rays.back().orig.x(), xmax, xmin, nx);

		for (auto & r2: b2.rays) {  // Rays in Beam 2
			auto p2_start = get_y_index(b1.rays.front().orig.y(), ymax, ymin, ny);
			auto p2_end = get_y_index(b1.rays.back().orig.y(), ymax, ymin, ny);

			for (int p1 = p1_start; p1 < p1_end; p1++) {   // Ray 1 points
				for (int p2 = p2_start; p2 < p2_end; p2++) {   // Ray 2 point
					auto ix = get_x_index(r1.path[p1].x(), xmax, xmin, nx);
					auto iz = get_y_index(r1.path[p1].y(), ymax, ymin, ny);

					auto ix2 = get_x_index(r2.path[p2].x(), xmax, xmin, nx);
					auto iz2 = get_y_index(r2.path[p2].y(), ymax, ymin, ny);

					if (ix == ix2 && iz == iz2) {   // Ray 1 and Ray 2 have same logical indices (same box)
						auto dk1 = r1.path[p1 + 1] - r1.path[p1];
						auto dk2 = r2.path[p2 + 1] - r2.path[p2];

						auto ne = eg.eden[iz][ix];
						auto epsilon = std::sqrt(1.0f - ne / ncrit);

						auto kmag = (omega / c) * epsilon;

						auto k1 = kmag * (dk1 / dk1.length());
						auto k2 = kmag * (dk2 / dk2.length());

						auto kiaw = std::sqrt(std::pow(k2[0] - k1[0], 2.0f) + std::pow(k2[1] - k2[1], 2.0f));
						auto ws = kiaw * cs;

						auto machnum = (((-0.4f) - (-2.4f)) / (xmax - xmin)) * (get_x_val(ix, xmax, xmin, nx) - xmin) + (-2.4f);
						auto u_flow = std::max(machnum, 0.0f) * cs;
						auto eta = ((omega - omega) - (k2[0] - k1[0]) * u_flow) / (ws + 1.0e-10f);   // omega is not changed in this code

						auto efield1 = std::sqrt(8.0f * pi * 1.0e7f * b1.edep[iz][ix] / c);
						auto P = (std::pow(iaw, 2.0f) * eta) / (std::pow(std::pow(eta, 2.0f) - 1.0f, 2.0f) + std::pow(iaw, 2.0f) * std::pow(eta, 2.0f));
						auto gain2 = constant1 * std::pow(efield1, 2.0f) * (ne / ncrit) * (1.0f / iaw) * P;

						eg.W_new[iz][ix][0] = eg.W[iz][ix][0] * std::exp(1 * eg.W[iz][ix][1] * dk1.length() * gain2 / epsilon);
						eg.W_new[iz][ix][1] = eg.W[iz][ix][1] * std::exp(-1 * eg.W[iz][ix][0] * dk2.length() * gain2 / epsilon);
					}
				}
			}
		}
	}
}

void calc_intensity(Beam& b1, Beam& b2, Egrid& eg) {
	for (auto& r1: b1.rays) {   // Rays in Beam 1
		auto p1_start = get_x_index(b2.rays.front().orig.x(), xmax, xmin, nx);
		auto p1_end = get_x_index(b2.rays.back().orig.x(), xmax, xmin, nx);

		for (auto & r2: b2.rays) {  // Rays in Beam 2
			auto p2_start = get_y_index(b1.rays.front().orig.y(), ymax, ymin, ny);
			auto p2_end = get_y_index(b1.rays.back().orig.y(), ymax, ymin, ny);

			for (int p1 = p1_start; p1 < p1_end; p1++) {   // Ray 1 points
				for (int p2 = p2_start; p2 < p2_end; p2++) {   // Ray 2 point
					auto ix = get_x_index(r1.path[p1].x(), xmax, xmin, nx);
					auto iz = get_y_index(r1.path[p1].y(), ymax, ymin, ny);

					auto ix2 = get_x_index(r2.path[p2].x(), xmax, xmin, nx);
					auto iz2 = get_y_index(r2.path[p2].y(), ymax, ymin, ny);

					if (ix == ix2 && iz == iz2) {   // Ray 1 and Ray 2 have same logical indices (same box)
						float frac_change_1 = -1.0f * (1.0f - (eg.W_new[iz][ix][0] / eg.W[iz][ix][0])) * b1.edep[iz][ix];
						float frac_change_2 = -1.0f * (1.0f - (eg.W_new[iz][ix][1] / eg.W[iz][ix][1])) * b2.edep[iz][ix];

						b1.edep_new[iz][ix] += frac_change_1;
						b2.edep_new[iz][ix] += frac_change_2;

						for (int q1 = p1 + 1; q1 < r1.path.size(); q1++) {
							auto ix_cur = get_x_index(r1.path[q1].x(), xmax, xmin, nx);
							auto iz_cur = get_y_index(r1.path[q1].y(), ymax, ymin, ny);

							if (ix_cur == ix || iz_cur == iz) { // Prevent double deposition in same zone
								b1.edep_new[iz_cur][ix_cur] += frac_change_1; // No correction for number of rays in zone;
							}

							ix = ix_cur;
							iz = iz_cur;
						}

						for (int q2 = p2 + 1; q2 < r2.path.size(); q2++) {
							auto ix_cur = get_x_index(r2.path[q2].x(), xmax, xmin, nx);
							auto iz_cur = get_y_index(r2.path[q2].y(), ymax, ymin, ny);

							if (ix_cur == ix || iz_cur == iz) { // Prevent double deposition in same zone
								b2.edep_new[iz_cur][ix_cur] += frac_change_1; // No correction for number of rays in zone;
							}

							ix = ix_cur;
							iz = iz_cur;
						}
					}
				}
			}
		}
	}
}

// Utility Functions
void save_beam_to_file(Beam& beam, const std::string& beam_name) {
	std::ofstream beam_file(beam_name + ".csv");
	beam_file << std::setprecision(std::numeric_limits<float>::max_digits10);

	for (int r = 0; r < beam.rays.size(); ++r) {
		for (int i = 0; i < beam.rays[r].path.size(); i++) {
			beam_file << r << ", " << beam.rays[r].path[i] << std::endl;
		}
	}
	beam_file.close();

	std::ofstream edep_file(beam_name + "_edep.csv");
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

	std::ofstream edep_new_file(beam_name + "_edep_new.csv");
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