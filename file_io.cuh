#ifndef CUCBET_FILE_IO_CUH
#define CUCBET_FILE_IO_CUH

#include "Beam.cuh"
#include "Egrid.cuh"

// Utility Functions
void save_beam_to_file(const Beam& beam, const std::string& beam_name) {
	const std::string output_path = "./Outputs/" + beam_name;

	// Write beam to file
	std::ofstream beam_file(output_path + ".csv");
	beam_file << std::setprecision(std::numeric_limits<float>::max_digits10);

	for (int r = 0; r < nrays; r++) {
		for (int t = 0; t < beam.rays[r].endex; t++) {
			beam_file << r << ", " << beam.rays[r].path[t] << "\n";
		}
	}
	beam_file.close();

	// Write beam edep to file (i_b#)
	std::ofstream edep_file(output_path + "_edep.csv");
	edep_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int y = ny - 1; y >= 0; y--) {
		for (int x = 0; x < nx; x++) {
			int index = y * nx + x;
			edep_file << beam.edep[index];
			if (x != nx - 1) {
				edep_file << ", ";
			}
		}
		edep_file << "\n";
	}
	edep_file.close();

	// Write beam edep_new to file (i_b#_new)
	std::ofstream edep_new_file(output_path + "_edep_new.csv");
	edep_new_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int y = ny - 1; y >= 0; y--) {
		for (int x = 0; x < nx; x++) {
			int index = y * nx + x;
			edep_new_file << beam.edep[index];
			if (x != nx - 1) {
				edep_new_file << ", ";
			}
		}
		edep_new_file << "\n";
	}
	edep_new_file.close();
}

void save_egrid_to_files(Egrid& eg) {
	const std::string output_path = "./Outputs/";

	// Write eden to file
	std::ofstream eden_file(output_path + "eden.csv");
	eden_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int i = ny - 1; i >= 0; i--) {
		for (int j = 0; j < nx; j++) {
			int index = i * nx + j;
			eden_file << eg.eden[index];
			if (j != nx - 1) {
				eden_file << ", ";
			}
		}
		eden_file << "\n";
	}
	eden_file.close();
}
#endif //CUCBET_FILE_IO_CUH
