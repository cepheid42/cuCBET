#include <iostream>
#include <fstream>
#include <vector>

#include "vec2.cuh"
#include "Beam.cuh"

int main() {
	float x_range = 10.0;
	float z_range = 10.0;
	float beam_width = 6.0;
	int nt = 20;
	int num_rays = 10;

	Vec beam1_dir(1.0, 0.0);
	Vec beam2_dir(0.0, 1.0);

	Beam b1(0, beam_width, num_rays, beam1_dir);
	Beam b2(1, beam_width, num_rays, beam2_dir);

	b1.init_beam(z_range, x_range, nt);
	b2.init_beam(x_range, z_range, nt);

	find_intersections(b1, b2);

	save_beam_to_file(b1, "beam1");
	save_beam_to_file(b2, "beam2");
	save_intersections(b1);

	return 0;
}
