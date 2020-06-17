#include "Beam.cuh"
#include "Egrid.cuh"
#include "Initializers.cuh"
#include "Intersections.cuh"
#include "file_io.cuh"
#include "Calculations.cuh"

using namespace std;

#include <chrono>
using chrono::steady_clock;
using chrono::duration_cast;
using chrono::duration;


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

	size_t heap_size;
	cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
	cout << "Setting GPU Heap size to " << heap_size / 1024.0 << " MB." << endl;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);

	// Electron Grid
	cout << "Initializing Electron Grid." << endl;
	Egrid *eg;
	checkErr(cudaMallocManaged(&eg, sizeof(Egrid)))
	eg->allocate();
	init_egrid(*eg);

	// Horizontal Beam
	auto beam1_start = steady_clock::now();
	cout << "Initializing Beam 1." << endl;
	Vec beam1_dir(1.0f, -0.1f);
	Beam *beam1;
	checkErr(cudaMallocManaged(&beam1, sizeof(Beam)))
	beam1->allocate(0, beam1_dir);

	float b1_x_start = xmin;
	float b1_y_start = beam_min - (dx / 2.0f);
	float b1_step = (beam_max - beam_min) / float(nrays - 1);
	init_beam(*beam1, *eg, b1_x_start, b1_y_start, b1_step);

	// Vertical Beam
	auto beam2_start = steady_clock::now();
	cout << "Initializing Beam 2." << endl;
	Vec beam2_dir(0.0f, 1.0f);
	Beam *beam2;
	checkErr(cudaMallocManaged(&beam2, sizeof(Beam)))
	beam2->allocate(1, beam2_dir);
	float b2_x_start = beam_min - (dy / 2.0f);
	float b2_y_start = ymin;
	float b2_step = (beam_max - beam_min) / float(nrays - 1);
	init_beam(*beam2, *eg, b2_x_start, b2_y_start, b2_step);


	// Find intersections
	auto crossing_start = steady_clock::now();
	cout << "Calculating Intersections (GPU)." << endl;
	get_intersections(*beam1, *beam2);

	// Calculate gain
	auto gain_start = steady_clock::now();
	cout << "Calculating Gain (GPU)." << endl;
	gpu_calc_gain(*beam1, *beam2, *eg);

	// Calculate intensity
	auto intensity_start = steady_clock::now();
//	calc_intensity(b1, b2, egrid, pm);
//
	auto io_start = steady_clock::now();
	cout << "Writing files." << endl;
	save_beam_to_file(*beam1, "beam1");
	save_beam_to_file(*beam2, "beam2");
	save_egrid_to_files(*eg);

	auto cleanup_start = steady_clock::now();
	cout << "Finished. Cleaning up...\n" << endl;
	checkErr(cudaFree(beam1))
	checkErr(cudaFree(beam2))
	checkErr(cudaFree(eg))

	// Time info
	auto grid_time      = duration_cast<duration<float>>(beam1_start         - prog_start);
	auto beam1_time     = duration_cast<duration<float>>(beam2_start         - beam1_start);
	auto beam2_time     = duration_cast<duration<float>>(crossing_start      - beam2_start);
	auto crossing_time  = duration_cast<duration<float>>(gain_start          - crossing_start);
	auto gain_time      = duration_cast<duration<float>>(intensity_start     - gain_start);
	auto intensity_time = duration_cast<duration<float>>(io_start            - intensity_start);
	auto io_time        = duration_cast<duration<float>>(cleanup_start       - io_start);
	auto cleanup_time   = duration_cast<duration<float>>(steady_clock::now() - cleanup_start);
	auto total_time     = duration_cast<duration<float>>(steady_clock::now() - prog_start);

	cout
	<< setw(25) << left << "Grid time: "         << fixed << setprecision(5) << left << grid_time.count()      << "s\n"
	<< setw(25) << left	<< "Beam 1 time: "       << fixed << setprecision(5) << left << beam1_time.count()     << "s\n"
	<< setw(25) << left << "Beam 2 time: "       << fixed << setprecision(5) << left << beam2_time.count()     << "s\n"
	<< setw(25) << left << "Intersection time: " << fixed << setprecision(5) << left << crossing_time.count()  << "s\n"
	<< setw(25) << left	<< "Gain time: "         << fixed << setprecision(5) << left << gain_time.count()      << "s\n"
	<< setw(25) << left	<< "Intensity time: "    << fixed << setprecision(5) << left << intensity_time.count() << "s\n"
	<< setw(25) << left	<< "File Write time: "   << fixed << setprecision(5) << left << io_time.count()        << "s\n"
	<< "====================================="   << endl
	<< setw(25) << left	<< "Total time: "        << fixed << setprecision(5) << left << total_time.count()     << "s" << endl;
	return 0;
}
