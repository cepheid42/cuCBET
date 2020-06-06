#include "Beam.cuh"
#include "Egrid.cuh"
//#include "Interpolator.cuh"

#include <chrono>
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;

/* Todo:
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


int main() {
	auto prog_start = steady_clock::now();

	auto* pm = new param_struct;

	auto grid_start = steady_clock::now();

	auto* egrid = new Egrid;

	egrid->allocate(pm);
	init_egrid(egrid, pm);

	// Horizontal beam
	auto beam1_start = steady_clock::now();
	Vec beam1_dir(1.0f, 0.1f);
	auto *beam1 = new Beam(0, pm->nrays, beam1_dir);
	beam1->allocate(pm);

	float b1_x_start = pm->xmin - (pm->dt / pm->courant_mult * pm->c * 0.5f);
	float b1_y_start = pm->beam_min - (pm->dy / 2.0f) - (pm->dt / pm->courant_mult * pm->c * 0.5f);
	float b1_y_step = (pm->beam_max - pm->beam_min) / float(pm->nrays - 1);
	init_beam(beam1, egrid, pm, b1_x_start, b1_y_start, b1_y_step);

	// Vertical beam
	auto beam2_start = steady_clock::now();
	Vec beam2_dir(0.0f, 1.0f);
	auto *beam2 = new Beam(1, pm->nrays, beam2_dir);
	beam2->allocate(pm);

	float b2_x_start = pm->beam_min - (pm->dx / 2.0f) - (pm->dt / pm->courant_mult * pm->c * 0.5f);
	float b2_x_step = (pm->beam_max - pm->beam_min) / float(pm->nrays - 1);
	float b2_y_start = pm->ymin - (pm->dt / pm->courant_mult * pm->c * 0.5f);
	init_beam(beam2, egrid, pm, b2_x_start, b2_y_start, b2_x_step);

//	// Find intersections
//	auto intersection_start = steady_clock::now();
//	find_intersections(b1, b2, pm);
//
//	// Calculate gain
//	auto gain_start = steady_clock::now();
//	calc_gain(b1, b2, egrid, pm);
//
//	// Calculate intensity
//	auto intensity_start = steady_clock::now();
//	calc_intensity(b1, b2, egrid, pm);

	auto file_write_start = steady_clock::now();
	save_beam_to_file(beam1, "beam1", pm);
	save_beam_to_file(beam2, "beam2", pm);
	save_egrid_to_files(egrid, pm);

	beam1->deallocate();
	delete beam1;

	beam2->deallocate();
	delete beam2;

	egrid->deallocate();
	delete egrid;

	delete pm;

//	// Time info
//	auto time_stop = steady_clock::now();
//	auto grid_time = duration_cast<duration<float>>(beam1_start - grid_start);
//	auto beam1_time = duration_cast<duration<float>>(beam2_start - beam1_start);
//	auto beam2_time = duration_cast<duration<float>>(intersection_start - beam2_start);
//	auto intersection_time = duration_cast<duration<float>>(gain_start - intersection_start);
//	auto gain_time = duration_cast<duration<float>>(intensity_start - gain_start);
//	auto intensity_time = duration_cast<duration<float>>(file_write_start - intensity_start);
//	auto file_write_time = duration_cast<duration<float>>(time_stop - file_write_start);
//	auto total_time = duration_cast<duration<float>>(steady_clock::now() - prog_start);
//
//	std::cout << "Grid time: "          << grid_time.count()            << " s\n"
//			  << "Beam 1 time: "        << beam1_time.count()           << " s\n"
//			  << "Beam 2 time: "        << beam2_time.count()           << " s\n"
//			  << "Intersection time: "  << intersection_time.count()    << " s\n"
//			  << "Gain time: "          << gain_time.count()            << " s\n"
//			  << "Intensity time: "     << intensity_time.count()       << " s\n"
//			  << "File Write time: "    << file_write_time.count()      << " s\n"
//			  << "Total time: "         << total_time.count()           << " s" << std::endl;
	return 0;
}
