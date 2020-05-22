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
		// Initialize e_dep matrix
		edep = new double*[nz + 2];
		for (int i = 0; i < nz + 2; ++i) {
			edep[i] = new double[nx + 2];
		}
	}

	~Beam() {
		for (int i = 0; i < nz + 2; ++i) {
			delete[] edep[i];
		}
		delete[] edep;
	}

public:
	int beam_num;
	int nrays;
	Vec direction;
	std::vector<Ray> rays;
	double **edep;          // (nz + 2, nx + 2), each beam tracks it's electron deposition
};

void init_beam(Beam& b, Egrid& eg, double x_start, double z_start, double step, double dt, int nt, double ncrit) {
	// Each ray can be initialized independently
	// So parallelize this entire function
	double phase_x = beam_min_z;
	double phase_step = (beam_max_z - beam_min_z) / (nrays - 1);
	double pow_x = exp(-1 * pow(pow(phase_x / sigma, 2.0), 2.0));

	for (int r = 0; r < b.nrays; ++r) {
		double power;
		if (b.beam_num == 0) {
			power = uray_mult * interp(b.direction.z(), phase_x + offset, pow_x);
		} else {
			power = uray_mult * interp(b.direction.x(), phase_x, pow_x);
		}

		Point ray_orig(x_start, z_start);
		Ray ray1(ray_orig, b.direction, power);
		draw_init_path(ray1, nt, dt, ncrit, eg, &(b.edep));
		b.rays.emplace_back(ray1);

		phase_x += phase_step;
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

	std::ofstream beam_edep(beam_name + "_edep.csv");
	save_2d_grid(beam_edep, beam.edep, nx, nz);
	beam_edep.close();
}

#endif //CUCBET_BEAM_CUH
