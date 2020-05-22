#include "vec2.cuh"
#include "Beam.cuh"
#include "Egrid.cuh"

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

using std::pow;

int main() {
	auto start = high_resolution_clock::now();

	double ncrit = 1e-6 * (pow(omega, 2.0) * m_e * e_0 / pow(e_c, 2.0));
	double dt = courant_mult * min(dx, dz) / c;
	auto nt = static_cast<int>(2 * max(nx, nz) / courant_mult);

	/* Todo:
	 * [] Replace all 2d array pointers with vector of vectors: double** a -> vector<vector<double>> a
	 * [] Figure out electron grid stuff
	 *      []  Each beam needs its own EDEP for intensity tracking
	 *      [x] Redo file output
	 *
	 * [] Add power weighting to rays (intensities and what not)
	 *
	 * [] Determine which code can be parallelized
	 * [] Put everything in unified memory for CUDA
	 * [] Make parameter struct or something
	 * [] Clean up dependencies
	 * [] Check for memory leaks
	*/

	Egrid egrid;
	init_eden_wpe(egrid, ncrit);
	init_eden_derivs(egrid);

	Vec beam1_dir(1.0, -0.1);
	Beam b1(0, nrays, beam1_dir);
	double b1_x_start = xmin - (dt / courant_mult * c * 0.5);
	double b1_z_start = beam_min_z + offset - (dz / 2) - (dt / courant_mult * c * 0.5);
	double b1_z_step = (beam_max_z - beam_min_z) / (nrays - 1);
	init_beam(b1, egrid, b1_x_start, b1_z_start, b1_z_step, dt, nt, ncrit);

	Vec beam2_dir(0.0, 1.0);
	Beam b2(1, nrays, beam2_dir);
	double b2_x_start = beam_min_z - (dx / 2) - (dt / courant_mult * c * 0.5);
	double b2_x_step = (beam_max_z - beam_min_z) / (nrays - 1);
	double b2_z_start = zmin - (dt / courant_mult * c * 0.5);
	init_beam(b2, egrid, b2_x_start, b2_z_start, b2_x_step, dt, nt, ncrit);

	save_beam_to_file(b1, "beam1");
	save_beam_to_file(b2, "beam2");
	save_egrid_to_files(egrid);

	auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	std::cout << "Time: " << duration.count() << " ms" << std::endl;
	return 0;
}
