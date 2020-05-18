#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include <vector>
#include <cassert>

#include "Ray.cuh"

class Beam {
public:
	Beam() = delete;
	Beam(int beam_id, float beam_width, int num_rays, Vec dir) :
		beam_num(beam_id),
		nrays(num_rays),
		width(beam_width)
	{
		direction = unit_vector(dir);
	}

	void init_beam(float range, float ray_len, int nt) {
		float start = (range - width) / 2.0f;
		float step = 0.5;

		for (int r = 0; r < nrays; ++r) {
			Point ray_orig;

			// x axis = 0, z axis = 1
			if (beam_num == 0) {
				ray_orig = Point(0.0, start);
			} else {
				ray_orig = Point(start, 0.0);
			}

			Ray ray1(ray_orig, direction);
			draw_init_path(ray1, ray_len, nt);
			rays.emplace_back(ray1);

			start += step;
		}
	}

public:
	int beam_num;
	int nrays;
	float width;
	Vec direction;
	std::vector<Ray> rays;
};

void find_intersections(Beam& b1, Beam& b2) {
	// Iterates over all rays in beam 1 and stores each intersection
	// in that ray class, so each ray has record of its own intersections.
	for (Ray& r1: b1.rays) {
		for (Ray& r2: b2.rays) {
			r1.ray_intersections(r2);
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

void save_intersections(Beam& beam) {
	std::ofstream myFile("intersections.csv");

	for (int ray_num = 0; ray_num < beam.rays.size(); ++ray_num) {
		for (auto i: beam.rays[ray_num].intersections) {
			myFile << ray_num << ", " << i << std::endl;
		}
	}

	myFile.close();
}

#endif //CUCBET_BEAM_CUH
