#include "Beam.cuh"
#include "Egrid.cuh"
#include "Intersections.cuh"
//#include "Calculations.cuh"
#include "file_io.cuh"

using namespace std;

int main() {
	timer prog_timer;
	prog_timer.start();

	int device = -1;
	cudaGetDevice(&device);

	timer grid_timer;
	timer beam1_timer;
	timer beam2_timer;
	timer xing_timer;
	timer gain_timer;
	timer intensity_timer;
	timer io_timer;
	timer cleanup_timer;

	// Electron Grid
	cout << "Initializing Electron Grid." << endl;
	grid_timer.start();

	Egrid *eg;
	checkErr(cudaMallocManaged(&eg, sizeof(Egrid)))
	eg->allocate();
	init_egrid(*eg);
	grid_timer.stop();

	// Horizontal Beam
	cout << "Initializing Beam 1." << endl;
	beam1_timer.start();

	Beam *beam1;
	checkErr(cudaMallocManaged(&beam1, sizeof(Beam)))
	beam1->allocate(0, 1.0f, -0.1f);
	float b1_x_start = xmin;
	float b1_y_start = beam_min - (dx / 2.0f);
	float b1_step = (beam_max - beam_min) / float(nrays - 1);
	init_beam(*beam1, *eg, b1_x_start, b1_y_start, b1_step);
	beam1_timer.stop();

	// Vertical Beam
	cout << "Initializing Beam 2." << endl;
	beam2_timer.start();

	Beam *beam2;
	checkErr(cudaMallocManaged(&beam2, sizeof(Beam)))
	beam2->allocate(1, 0.0f, 1.0f);
	float b2_x_start = beam_min - (dy / 2.0f);
	float b2_y_start = ymin;
	float b2_step = (beam_max - beam_min) / float(nrays - 1);
	init_beam(*beam2, *eg, b2_x_start, b2_y_start, b2_step);
	beam2_timer.stop();

	// Find intersections
	cout << "Calculating Intersections (GPU)." << endl;
	xing_timer.start();
	get_intersections(*beam1, *beam2);
	xing_timer.stop();

	// Calculate gain
//	cout << "Calculating Gain (GPU)." << endl;
	gain_timer.start();
//	gpu_calc_gain(*beam1, *beam2, *eg);
	gain_timer.stop();

	// Calculate intensity
//	cout << "Calculating Gain (GPU)." << endl;
	intensity_timer.start();
//	calc_intensity(b1, b2, egrid, pm);
	intensity_timer.stop();

	// Save to file
	cout << "Writing files." << endl;
	io_timer.start();
	save_beam_to_file(*beam1, "beam1");
	save_beam_to_file(*beam2, "beam2");
	save_egrid_to_files(*eg);
	io_timer.stop();

	// Cleanup
	cout << "Finished. Cleaning up...\n" << endl;
	cleanup_timer.start();

	beam1->free_mem();
	beam2->free_mem();
	eg->free_mem();

	checkErr(cudaFree(beam1))
	checkErr(cudaFree(beam2))
	checkErr(cudaFree(eg))

	cleanup_timer.stop();
	prog_timer.stop();

	cout << "=================================\n"
	<< setw(25) << left << "Grid time: "         << fixed << setprecision(5) << left << grid_timer.elapsed      << "s\n"
	<< setw(25) << left	<< "Beam 1 time: "       << fixed << setprecision(5) << left << beam1_timer.elapsed     << "s\n"
	<< setw(25) << left << "Beam 2 time: "       << fixed << setprecision(5) << left << beam2_timer.elapsed     << "s\n"
	<< setw(25) << left << "Intersection time: " << fixed << setprecision(5) << left << xing_timer.elapsed      << "s\n"
	<< setw(25) << left	<< "Gain time: "         << fixed << setprecision(5) << left << gain_timer.elapsed      << "s\n"
	<< setw(25) << left	<< "Intensity time: "    << fixed << setprecision(5) << left << intensity_timer.elapsed << "s\n"
	<< setw(25) << left	<< "File Write time: "   << fixed << setprecision(5) << left << io_timer.elapsed        << "s\n"
	<< "================================="   << '\n'
	<< setw(25) << left	<< "Total time: "        << fixed << setprecision(5) << left << prog_timer.elapsed      << "s"
	<< endl;
	return 0;
}
