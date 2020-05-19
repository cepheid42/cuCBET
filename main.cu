#include "vec2.cuh"
#include "Beam.cuh"
//#include "Egrid.cuh"

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

using std::pow;

int main() {
	auto start = high_resolution_clock::now();

	auto ncrit = m_e * pow(omega, 2.0)/ (4 * pi * pow(e_c, 2));
	double dt = courant_mult * std::min(dx, dz) / c;
	auto nt = static_cast<int>(2 * std::max(nx, nz) / courant_mult);


	Vec beam1_dir(1.0, 0.0);
	Vec beam2_dir(0.0, 1.0);

	Beam b1(0, beam_min_z, beam_max_z, nrays, beam1_dir);
	Beam b2(1, beam_min_z, beam_max_z, nrays, beam2_dir);

	double b1_x_start = xmin - (dt / courant_mult * c * 0.5);
	double b1_z_start = beam_min_z + offset - (dz / 2) - (dt / courant_mult * c * 0.5);
	double b1_z_step = (beam_max_z - beam_min_z) / (nrays - 1);
	init_beam(b1, b1_x_start, b1_z_start, b1_z_step, dt, nt, ncrit);

	double b2_x_start = beam_min_z - (dx / 2) - (dt / courant_mult * c * 0.5);
	double b2_x_step = (beam_max_z - beam_min_z) / (nrays - 1);
	double b2_z_start = zmin - (dt / courant_mult * c * 0.5);
	init_beam(b2, b2_x_start, b2_z_start, b2_x_step, dt, nt, ncrit);

	save_beam_to_file(b1, "beam1");
	save_beam_to_file(b2, "beam2");

	auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	std::cout << "Time: " << duration.count() << " ms" << std::endl;
	return 0;
}
