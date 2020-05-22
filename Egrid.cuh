#ifndef CUCBET_EGRID_CUH
#define CUCBET_EGRID_CUH

#include "vec2.cuh"

class Egrid {
public:
	Egrid() {
		// Initialize e_den, derivatives, and wpe
		eden = new double*[nz];
		d_eden = new Point*[nz];
		wpe = new double*[nz];
		for (int i = 0; i < nz; ++i) {
			eden[i] = new double[nx];
			d_eden[i] = new Point[nx];
			wpe[i] = new double[nx];
		}


	}
	~Egrid() {
		for (int i = 0; i < nz; ++i) {
			delete[] eden[i];
			delete[] d_eden[i];
			delete[] wpe[i];
		}
		delete[] eden;
		delete[] d_eden;
		delete[] wpe;
	}

public:
	double **eden;          // (nz, nx)
	Point **d_eden;         // (nz, nx)
	double **wpe;
};

void init_eden_wpe(Egrid& eg, double ncrit) {
	// Todo: This does not output exactly the same as Yorick code, but is close (same order of mag, 9.12345xxxxxxxxxxxE+20)
	for (int i = 0; i < nz; ++i) {
		for (int j = 0; j < nx; ++j) {
			double density = ((0.3 * ncrit - 0.1 * ncrit) / (xmax - xmin)) * (get_grid_val(j, xmax, xmin, nx) - xmin) + (0.1 * ncrit);
			eg.eden[i][j] = max(density, 0.0);

			double plasma_freq = sqrt(eg.eden[i][j] * 1.0e6 * pow(e_c, 2.0) / (m_e * e_0));
			eg.wpe[i][j] = plasma_freq;
		}
	}
}

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


// Utility Functions
void output_d_eden_to_files(Point** deriv_grid, int n_x, int n_z) {
	std::ofstream dx_file("d_eden_dx.csv");
	std::ofstream dz_file("d_eden_dz.csv");

	dx_file << std::setprecision(std::numeric_limits<double>::max_digits10);
	dz_file << std::setprecision(std::numeric_limits<double>::max_digits10);

	for (int i = 0; i < n_z; ++i) {
		for (int j = 0; j < n_x; ++j) {
			dx_file << deriv_grid[i][j][0];
			dz_file << deriv_grid[i][j][1];

			if (j != n_x - 1) {
				dx_file << ", ";
				dz_file << ", ";
			}
		}
		dx_file << "\n";
		dz_file << "\n";
	}
	dx_file.close();
	dz_file.close();
}

void save_egrid_to_files(Egrid& eg) {
	// Maybe parallelize this on CPU, if grids get very big or something
	std::ofstream save_eden("eden.csv");
	save_2d_grid(save_eden, eg.eden, nx, nz);
	save_eden.close();

	output_d_eden_to_files(eg.d_eden, nx, nz);
}

#endif //CUCBET_EGRID_CUH
