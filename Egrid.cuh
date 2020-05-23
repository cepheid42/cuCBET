#ifndef CUCBET_EGRID_CUH
#define CUCBET_EGRID_CUH

#include "vec2.cuh"

using std::array;

class Egrid {
public:
	Egrid() = default;

public:
	array<array<double, nx>, nz> eden{};
	array<array<Point, nx>, nz> d_eden;
	array<array<Point, nx>, nz> W;
	array<array<Point, nx>, nz> W_new;
	std::array<std::array<double, nx>, nz> present{};
};

void init_eden_derivs(Egrid& eg) {
	// This function may possible be parallelizable...
	for (int i = 0; i < nz - 1; ++i) {
		for (int j = 0; j < nx - 1; ++j) {
			eg.d_eden[i][j][0] = (eg.eden[i][j + 1] - eg.eden[i][j]) / (get_grid_val(j + 1, xmax, xmin, nx) - get_grid_val(j, xmax, xmin, nx));
			eg.d_eden[i][j][1] = (eg.eden[i + 1][j] - eg.eden[i][j]) / (get_grid_val(i + 1, zmax, zmin, nz) - get_grid_val(i, zmax, zmin, nz));
		}
	}

	// set last column equal to previous column
	for (int i = 0; i < nz - 1; ++i) {
		eg.d_eden[nx - 1][i] = Point(eg.d_eden[nx - 2][i]);
	}
}

void init_egrid(Egrid& eg, double ncrit) {
	// Todo: This does not output exactly the same as Yorick code, but is close (same order of mag, 9.12345xxxxxxxxxxxE+20)
	for (int i = 0; i < nz; ++i) {
		for (int j = 0; j < nx; ++j) {
			double density = ((0.3 * ncrit - 0.1 * ncrit) / (xmax - xmin)) * (get_grid_val(j, xmax, xmin, nx) - xmin) + (0.1 * ncrit);
			eg.eden[i][j] = max(density, 0.0);

			double w = sqrt(1 - eg.eden[i][j] / ncrit) / rays_per_zone;
			eg.W[i][j] = Point(w, w);
		}
	}
	eg.W_new = eg.W;
	init_eden_derivs(eg);
}

// Utility Functions
void save_egrid_to_files(Egrid& eg) {
	std::ofstream eden_file("eden.csv");
	eden_file << std::setprecision(std::numeric_limits<double>::max_digits10);
	for (auto & row : eg.eden) {
		for (auto &col: row) {
			eden_file << col << ", ";
		}
		eden_file << "\n";
	}
	eden_file.close();

	std::ofstream i_b1_file("i_b1.csv");
	std::ofstream i_b2_file("i_b2.csv");
	i_b1_file << std::setprecision(std::numeric_limits<double>::max_digits10);
	i_b2_file << std::setprecision(std::numeric_limits<double>::max_digits10);

	for (int i = 0; i < nz; i++) {
		for (int j = 0; j < nx; j++) {
			i_b1_file << eg.W[i][j][0] << ", ";
			i_b2_file << eg.W[i][j][1] << ", ";
		}
		i_b1_file << "\n";
		i_b2_file << "\n";
	}
	i_b1_file.close();
	i_b2_file.close();

	std::ofstream i_b1_new_file("i_b1_new.csv");
	std::ofstream i_b2_new_file("i_b2_new.csv");
	i_b1_new_file << std::setprecision(std::numeric_limits<double>::max_digits10);
	i_b2_new_file << std::setprecision(std::numeric_limits<double>::max_digits10);

	for (int i = 0; i < nz; i++) {
		for (int j = 0; j < nx; j++) {
			i_b1_new_file << eg.W_new[i][j][0] << ", ";
			i_b2_new_file << eg.W_new[i][j][1] << ", ";
		}
		i_b1_new_file << "\n";
		i_b2_new_file << "\n";
	}
	i_b1_new_file.close();
	i_b2_new_file.close();
}

#endif //CUCBET_EGRID_CUH
