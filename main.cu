#include "vec2.cuh"
#include "Beam.cuh"
//#include "Egrid.cuh"

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

int main() {
	auto start = high_resolution_clock::now();

	float x_range = 100.0;
	float z_range = 100.0;
	int nx = 200.0;
	float beam_width = 60.0;
	int nt = 200;
	int num_rays = 100;

	Vec beam1_dir(1.0, 0.0);
	Vec beam2_dir(0.0, 1.0);

	Beam b1(0, beam_width, num_rays, beam1_dir);
	Beam b2(1, beam_width, num_rays, beam2_dir);

	init_beam(b1, z_range, nx, nt);
	init_beam(b2, x_range, nx, nt);

//	find_intersections(b1, b2);

	save_beam_to_file(b1, "beam1");
	save_beam_to_file(b2, "beam2");

//	save_intersections(b1);

	auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	std::cout << "Time: " << duration.count() << std::endl;
	return 0;
}
