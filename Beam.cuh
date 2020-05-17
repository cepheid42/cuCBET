#ifndef CUCBET_BEAM_CUH
#define CUCBET_BEAM_CUH

#include <vector>
#include <cassert>

#include "Ray.cuh"

class Beam {
public:
	Beam() = default;
	Beam(int b, float beam_width, int num_rays, Vec dir) :
		beam_num(b),
		nrays(num_rays),
		width(beam_width),
		direction(dir)
	{
		assert(direction == unit_vector(direction));    // Check direction is unit vector
	}

	void init_beam(float x_range, int axis) {
		auto step = static_cast<int>(width / float(nrays - 1));     // Not final step size, nrays - 1?
		float start = (x_range - width) / 2;                        // Not final start location

		for (int r = 0; r < nrays; ++r) {
			Point ray_orig;

			// x axis = 0, z axis = 1
			if (axis == 0) {
				ray_orig = Point(start, 0.0);               // Not final ray origin
			} else {
				ray_orig = Point(0.0, start);
			}

			Ray ray1(ray_orig, direction);
			draw_init_path(ray1);
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

// Utility Functions
void save_beam_to_file(Beam beam, const std::string& beam_name) {
	std::ofstream myFile(beam_name + ".csv");

	for (int ray_num = 0; ray_num < beam.rays.size(); ++ray_num) {
		for (auto j : beam.rays[ray_num].path) {
			myFile << ray_num << ", " << j.x() << ", " << j.z() << std::endl;
		}
	}

	myFile.close();
}

#endif //CUCBET_BEAM_CUH
