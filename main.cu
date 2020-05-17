#include <iostream>
#include <fstream>
#include <vector>

#include "constants.cuh"
#include "vec2.cuh"
#include "Ray.cuh"
#include "Beam.cuh"


int main() {
	float x_range = 10.0;
	float z_range = 10.0;

	float beam_width = 5.0;
	int num_rays = 25;
	Vec beam1_orig(1.0, 0.0);
	Vec beam2_orig(0.0, 1.0);

	Beam b1(0, beam_width, num_rays, beam1_orig);
	Beam b2(1, beam_width, num_rays, beam2_orig);

	b1.init_beam(z_range, 0);
	b2.init_beam(x_range, 1);

	return 0;
}
