#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include "Ray.cuh"

class Beam {
public:
	Beam() = delete;
	Beam(int beam_id, int num_rays, const Vec& dir) :
		beam_num(beam_id),
		nrays(num_rays),
		direction(unit_vector(dir))
	{
		// Pre-size ray vector
		rays.reserve(nrays);
	}


public:
	int beam_num;
	int nrays;
	Vec direction;
	std::vector<Ray> rays;
	std::array<std::array<double, nx + 2>, nz + 2> edep{};  // Each beam tracks it's electron deposition
	std::array<std::array<double, nx + 2>, nz + 2> edep_new{};
};

void init_beam(Beam& b, Egrid& eg, double x_start, double z_start, double step, double dt, int nt, double ncrit) {
	// Each ray can be initialized independently
	// So parallelize this entire function
	for (int r = 0; r < b.nrays; ++r) {
		// Does phase_x1 have valid value when r + 1 > nrays?
		double phase_x0 = get_grid_val(r, beam_max_z, beam_min_z, nrays);
		double phase_x1 = get_grid_val(r + 1, beam_max_z, beam_min_z, nrays);

		double pow_x0 = exp(-1 * pow(pow(phase_x0 / sigma, 2.0), 2.0));
		double pow_x1 = exp(-1 * pow(pow(phase_x1 / sigma, 2.0), 2.0));

		double uray0;
		if (b.beam_num == 0) {
			uray0 = uray_mult * interp(z_start, phase_x0 + offset, pow_x0, phase_x1 + offset, pow_x1);
		} else {
			uray0 = uray_mult * interp(x_start, phase_x0, pow_x0, phase_x1, pow_x1);
		}

		Point ray_orig(x_start, z_start);
		Ray ray1(ray_orig, b.direction, uray0, nt);
		draw_init_path(ray1, nt, dt, ncrit, eg, b.edep);
		b.rays.emplace_back(ray1);

		if (b.beam_num == 0) {
			z_start += step;
		} else {
			x_start += step;
		}
	}
}


// Utility Functions
void save_beam_to_file(Beam& beam, const std::string& beam_name) {
	std::ofstream beam_file(beam_name + ".csv");
	beam_file << std::setprecision(std::numeric_limits<double>::max_digits10);
	for (int ray_num = 0; ray_num < beam.rays.size(); ++ray_num) {
		for (const auto& j : beam.rays[ray_num].path) {
			beam_file << ray_num << ", " << j << std::endl;
		}
	}
	beam_file.close();

	std::ofstream edep_file(beam_name + "_edep.csv");
	edep_file << std::setprecision(std::numeric_limits<double>::max_digits10);
	for (auto &row : beam.edep) {
		for (auto &col: row) {
			edep_file << col << ", ";
		}
		edep_file << "\n";
	}
	edep_file.close();
}

#endif //CUCBET_BEAM_CUH
