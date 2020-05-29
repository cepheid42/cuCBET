#include "vec2.cuh"
#include "Beam.cuh"
#include "Egrid.cuh"
#include "Interpolator.cuh"

#include <chrono>
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;


int main() {
	auto prog_start = steady_clock::now();

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
	 * [x] Scrub Z-X axis to X-Y axis, make everything consistent.
	 *
	 * [ ] Figure out gain calculation problems
	 *
	 * [ ] Intersection tracking
	 *      - Is it more efficient than brute force iteration thats currently implemented?
	 *      - Can brute force iteration be parallelized or offloaded to GPU?
	 *
	 *      - Rays per zone needed?
	 *
	 * [x] Replace all 2d array pointers with array of arrays: float** a -> array<array<float, nx>, nz> a
	 * [x] Figure out electron grid stuff
	 *      [x] Each beam needs its own EDEP for intensity tracking
	 *      [x] Redo file output
	 *
	 * [x] Add power weighting to rays (intensities and what not)
	 *      [x] Check electron deposition (make sure it's depositing in correct bins)
	 *
	 * [ ] Determine which code can be parallelized
	 * [ ] Put everything in unified memory for CUDA
	 * [ ] Clean up constants and parameters
	 *      [ ] Make parameter struct or something
	 *      [ ] global variables and constants, reduce and streamline
	 *          - Which ones are wanted for modifying? Would local declarations be better?
	 *
	 * [ ] Clean up dependencies
	 * [ ] Check for memory leaks
	 * [ ] Document code...
	 * [ ] Optimize
	 *      [ ] use vector Reserve method eg. vec.reserve(100)
	 *      [ ] verify row major operations (been playing fast and loose here)
	 *
	 * [ ] Implement function and GPU timing.
	*/

	/* Todo: Questions
	 * [ ] Group velocity is updated using d_eden[this_z0, this_x0] for every time step
	 *     should this be updated to be the d_eden[cur_z, cur_x]?
	 *
	 * [ ] eden and d_eden are never updated, would results be more accurate if they were
	 *     updated with deposition values at some point? AKA, iterative updates.
	 *
	 * [ ] Are CGS units needed? Would it be faster if all units were CGS (better range for floats?)
	*/

	auto grid_start = steady_clock::now();

	Egrid egrid;
	init_egrid(egrid, ncrit);

	auto beam1_start = steady_clock::now();
	auto grid_time = duration_cast<duration<float>>(beam1_start - grid_start);
	std::cout << "Grid time: " << grid_time.count() << " s" << std::endl;

	// Horizontal beam
	Vec beam1_dir(1.0f, 0.1f);
	Beam b1(0, nrays, beam1_dir);
	float b1_x_start = xmin - (dt / courant_mult * c * 0.5f);
	float b1_y_start = beam_min - (dy / 2.0f) - (dt / courant_mult * c * 0.5f);
	float b1_y_step = (beam_max - beam_min) / float(nrays - 1);
	init_beam(b1, egrid, b1_x_start, b1_y_start, b1_y_step, dt, nt, ncrit, phase_interp);

	auto beam2_start = steady_clock::now();
	auto beam1_time = duration_cast<duration<float>>(beam2_start - beam1_start);
	std::cout << "Beam 1 time: " << beam1_time.count() << " s" << std::endl;

	// Vertical beam
	Vec beam2_dir(0.0f, 1.0f);
	Beam b2(1, nrays, beam2_dir);
	float b2_x_start = beam_min - (dx / 2.0f) - (dt / courant_mult * c * 0.5f);
	float b2_x_step = (beam_max - beam_min) / float(nrays - 1);
	float b2_y_start = ymin - (dt / courant_mult * c * 0.5f);
	init_beam(b2, egrid, b2_x_start, b2_y_start, b2_x_step, dt, nt, ncrit, phase_interp);

	auto intersection_start = steady_clock::now();
	auto beam2_time = duration_cast<duration<float>>(intersection_start - beam2_start);
	std::cout << "Beam 2 time: " << beam2_time.count() << " s" << std::endl;

	find_intersections(b1, b2, dt);

	auto gain_start = steady_clock::now();
	auto intersection_time = duration_cast<duration<float>>(gain_start - intersection_start);
	std::cout << "Intersection time: " << intersection_time.count() << " s" << std::endl;
	calc_gain(b1, b2, egrid, ncrit);

	auto intensity_start = steady_clock::now();
	auto gain_time = duration_cast<duration<float>>(intensity_start - gain_start);
	std::cout << "Gain time: " << gain_time.count() << " s" << std::endl;

	calc_intensity(b1, b2, egrid);

	auto file_write_start = steady_clock::now();
	auto intensity_time = duration_cast<duration<float>>(file_write_start - intensity_start);
	std::cout << "Intensity time: " << intensity_time.count() << " s" << std::endl;

	save_beam_to_file(b1, "beam1");
	save_beam_to_file(b2, "beam2");
	save_egrid_to_files(egrid);

	auto file_write_time = duration_cast<duration<float>>(steady_clock::now() - file_write_start);
	std::cout << "File Write time: " << file_write_time.count() << " s" << std::endl;

	auto total_time = duration_cast<duration<float>>(steady_clock::now() - prog_start);
	std::cout << "Total time: " << total_time.count() << " s" << std::endl;
	return 0;
}
