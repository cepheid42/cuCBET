#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include "Ray.cuh"

class Beam {
public:
	Beam() = delete;
	Beam(int beam_id, double b_min, double b_max, int num_rays, Vec dir) :
		beam_num(beam_id),
		nrays(num_rays),
		beam_min(b_min),
		beam_max(b_max),
		direction(unit_vector(dir))
	{}

public:
	int beam_num;
	int nrays;
	double beam_min;
	double beam_max;
	Vec direction;
	std::vector<Ray> rays;
};

void init_beam(Beam& b, double x_start, double z_start, double step, double dt, int nt, double ncrit) {
	for (int r = 0; r < b.nrays; ++r) {
		Point ray_orig(x_start, z_start);

		Ray ray1(ray_orig, b.direction);
		draw_init_path(ray1, nt, dt, ncrit);
		b.rays.emplace_back(ray1);

		if (b.beam_num == 0) {
			z_start += step;
		} else {
			x_start += step;
		}
	}
}

//void find_intersections(Beam& b1, Beam& b2) {
//	for (Ray& r1: b1.rays) {
//		for (Ray& r2: b2.rays) {
//			ray_intersections(r1, r2);
//		}
//	}
//}


// Utility Functions
void save_beam_to_file(Beam& beam, const std::string& beam_name) {
	std::ofstream myFile(beam_name + ".csv");
	myFile << std::setprecision(std::numeric_limits<double>::max_digits10);

	for (int ray_num = 0; ray_num < beam.rays.size(); ++ray_num) {
		for (auto j : beam.rays[ray_num].path) {
			myFile << ray_num << ", " << j << std::endl;
		}
	}

	myFile.close();
}

//inline void save_intersections(Beam& beam) {
//	save_beam_to_file(beam, "intersections");
//}

#endif //CUCBET_BEAM_CUH
