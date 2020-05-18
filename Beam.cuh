#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include <vector>
#include <fstream>

#include "Ray.cuh"

class Beam {
public:
	Beam() = delete;
	Beam(int beam_id, float beam_width, int num_rays, Vec dir) :
		beam_num(beam_id),
		nrays(num_rays),
		width(beam_width),
		direction(unit_vector(dir))
	{}

public:
	int beam_num;
	int nrays;
	float width;
	Vec direction;
	std::vector<Ray> rays;
};

void init_beam(Beam& b, float range, int nx, int nt) {
	float start = (range - b.width) / 2.0f;
	float step = range / float(nx - 1);

	for (int r = 0; r < b.nrays; ++r) {
		Point ray_orig;

		if (b.beam_num == 0) {
			ray_orig = Point(0.0, start);
		} else {
			ray_orig = Point(start, 0.0);
		}

		Ray ray1(ray_orig, b.direction);
		draw_init_path(ray1, range, nt);
		b.rays.emplace_back(ray1);

		start += step;
	}
}

void find_intersections(Beam& b1, Beam& b2) {
	for (Ray& r1: b1.rays) {
		for (Ray& r2: b2.rays) {
			ray_intersections(r1, r2);
		}
	}
}


// Utility Functions
void save_beam_to_file(Beam& beam, const std::string& beam_name) {
	std::ofstream myFile(beam_name + ".csv");

	for (int ray_num = 0; ray_num < beam.rays.size(); ++ray_num) {
		for (auto j : beam.rays[ray_num].path) {
			myFile << ray_num << ", " << j << std::endl;
		}
	}

	myFile.close();
}

inline void save_intersections(Beam& beam) {
	save_beam_to_file(beam, "intersections");
}

#endif //CUCBET_BEAM_CUH
