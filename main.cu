#include "vec2.cuh"
#include "Beam.cuh"
#include "Egrid.cuh"
#include "Interpolator.cuh"

#include <chrono>
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;


int main() {
	auto start = steady_clock::now();

	float ncrit = 1e-6f * (std::pow(omega, 2.0f) * m_e * e_0 / std::pow(e_c, 2.0f));
	float dt = courant_mult * std::min(dx, dy) / c;
	auto nt = static_cast<int>(2.0f * std::max(nx, ny) / courant_mult);

	std::vector<std::pair<float, float>> phase_power;
	for (int i = 0; i < nrays; i++) {
		auto phase = get_x_val(i, beam_max, beam_min, nrays);
		auto power = std::exp(-1.0f * std::pow(std::pow(phase / sigma, 2.0f), 2.0f));

		phase_power.emplace_back(std::pair<float, float>(phase, power));
	}
	Interpolator phase_interp(phase_power);


	/* Todo:
	 * [ ] Scrub Z-X axis to X-Y axis, make everything consistent.
	 *      - vec2 class is (x, y) so make axis xy axis.
	 *
	 * [ ] Figure out gain calculation problems
	 *      [ ] Think it through, don't follow the old version. Old version is garbage
	 *      [ ] Not reaching print statements
	 *      [ ] Not updating values correctly
	 *
	 * [x] Replace all 2d array pointers with array of arrays: float** a -> array<array<float, nx>, nz> a
	 * [x] Figure out electron grid stuff
	 *      [x] Each beam needs its own EDEP for intensity tracking
	 *      [x] Redo file output
	 *
	 * [ ] Add power weighting to rays (intensities and what not)
	 *      [ ] Check electron deposition (make sure it's depositing in correct bins)
	 *
	 * [ ] Determine which code can be parallelized
	 * [ ] Put everything in unified memory for CUDA
	 * [ ] Clean up constants and parameters
	 *      [ ] Make parameter struct or something
	 *
	 * [ ] Clean up dependencies
	 * [ ] Check for memory leaks
	 * [ ] Document code...
	 * [ ] Optimize
	 *      [ ] use vector Reserve method eg. vec.reserve(100)
	 *      [ ] verify row major operations (been playing fast and loose here)
	*/

	/* Todo: Questions
	 * [ ] Group velocity is updated using d_eden[this_z0, this_x0] for every time step
	 *     should this be updated to be the d_eden[cur_z, cur_x]?
	*/

	Egrid egrid;
	init_egrid(egrid, ncrit);

	// Horizontal beam
	Vec beam1_dir(1.0f, 0.1f);
	Beam b1(0, nrays, beam1_dir);
	float b1_x_start = xmin - (dt / courant_mult * c * 0.5f);
	float b1_y_start = beam_min - (dy / 2.0f) - (dt / courant_mult * c * 0.5f);
	float b1_y_step = (beam_max - beam_min) / float(nrays - 1);
	init_beam(b1, egrid, b1_x_start, b1_y_start, b1_y_step, dt, nt, ncrit, phase_interp);

	// Vertical beam
	Vec beam2_dir(0.0f, 1.0f);
	Beam b2(1, nrays, beam2_dir);
	float b2_x_start = beam_min - (dx / 2.0f) - (dt / courant_mult * c * 0.5f);
	float b2_x_step = (beam_max - beam_min) / float(nrays - 1);
	float b2_y_start = ymin - (dt / courant_mult * c * 0.5f);
	init_beam(b2, egrid, b2_x_start, b2_y_start, b2_x_step, dt, nt, ncrit, phase_interp);

	calc_gain(b1, b2, egrid, ncrit);
	calc_intensity(b1, b2, egrid);

	save_beam_to_file(b1, "beam1");
	save_beam_to_file(b2, "beam2");
	save_egrid_to_files(egrid);

	duration<float> t1 = duration_cast<duration<float>>(steady_clock::now() - start);
	std::cout << "Time: " << t1.count() << " s" << std::endl;
	return 0;
}
