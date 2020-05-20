#ifndef CUCBET_EGRID_CUH
#define CUCBET_EGRID_CUH

#include "vec2.cuh"

class Egrid {
public:
	Egrid() {
		// Initialize e_den
		eden = new double*[nx];
		for (int i = 0; i < nx; ++i) {
			eden[i] = new double[nz];
		}
		// Initialize e_dep
		edep = new double*[nx + 2];
		for (int i = 0; i < nx + 2; ++i) {
			edep[i] = new double[nz + 2];
		}

		d_eden = new Point*[nx];
		for (int i = 0; i < nx; ++i) {
			d_eden[i] = new Point[nz];
		}

//		// Initialize d_eden_dx
//		d_eden_dx = new double*[nx];
//		for (int i = 0; i < nx; ++i) {
//			d_eden_dx[i] = new double[nz];
//		}
//		// Initialize d_eden_dz
//		d_eden_dz = new double*[nx];
//		for (int i = 0; i < nx; ++i) {
//			d_eden_dz[i] = new double[nz];
//		}
	}

	~Egrid() {
		for (int i = 0; i < nx; ++i) {
			delete[] eden[i];
			delete[] d_eden[i];
//			delete[] d_eden_dx[i];
//			delete[] d_eden_dz[i];
		}

		delete[] eden;
		delete[] d_eden;
//		delete[] d_eden_dx;
//		delete[] d_eden_dz;

		for (int i = 0; i < nx + 2; ++i) {
			delete[] edep[i];
		}

		delete[] edep;
	}

public:
	double **eden;          // (nx, nz)
	double **edep;          // (nx + 2, nz + 2)
	Point **d_eden;
//	double **d_eden_dx;     // (nx, nz)
//	double **d_eden_dz;     // (nx, nz)
};


inline double get_x_grid_val(int i) {
	return xmin + (i * (xmax - xmin) / (nx - 1));
}

inline double get_z_grid_val(int j) {
	return zmin + (j * (zmax - zmin) / (nz - 1));
}

inline int get_x_index(double x) {
	return static_cast<int>((nx - 1) * (x - xmin) / (xmax - xmin));
}

inline int get_z_index(double z) {
	return static_cast<int>((nz - 1) * (z - zmin) / (zmax - zmin));
}


void init_eden(Egrid& e, double ncrit) {
	// Todo: This does not output exactly the same as Yorick code, but is close (same order of mag, 9.12345xxxxxxxxxxxE+20)
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < nz; ++j) {
			double val = ((0.3 * ncrit - 0.1 * ncrit) / (xmax - xmin)) * (get_x_grid_val(i) - xmin) + (0.1 * ncrit);
			e.eden[i][j] = max(val, 0.0);
		}
	}
}

void init_eden_derivs(Egrid& e) {
	// This function may possible be parallelizable...
	for (int i = 0; i < nx - 1; ++i) {
		for (int j = 0; j < nz - 1; ++j) {
			e.d_eden[i][j][0] = (e.eden[i + 1][j] - e.eden[i][j]) / (get_x_grid_val(i + 1) - get_x_grid_val(i));
		}
	}

	for (int i = 0; i < nz - 1; ++i) {
		e.d_eden[nx - 1][i] = Point(e.d_eden[nx - 2][i]);
	}
}


// Utility Functions
void grid_out(std::ofstream& file, double** grid, int n_x, int n_z){
	file << std::setprecision(std::numeric_limits<double>::max_digits10);

	for (int i = 0; i < n_x; ++i) {
		for (int j = 0; j < n_z; ++j) {
			file << grid[i][j];
			if (j != n_z - 1) {
				file << ", ";
			}
		}
		file << "\n";
	}
}

void save_egrid_to_files(Egrid& e) {
	// Maybe parallelize this on CPU, if grids get very big or something
	std::ofstream save_eden("eden.csv");
	grid_out(save_eden, e.eden, nx, nz);
	save_eden.close();

	std::ofstream save_d_eden_dx("d_eden_dx.csv");
	grid_out(save_d_eden_dx, e.d_eden_dx, nx, nz);
	save_d_eden_dx.close();

	std::ofstream save_d_eden_dz("d_eden_dz.csv");
	grid_out(save_d_eden_dz, e.d_eden_dz, nx , nz);
	save_d_eden_dz.close();

	std::ofstream save_edep("edep.csv");
	grid_out(save_edep, e.edep, nx + 2, nz + 2);
	save_edep.close();
}


#endif //CUCBET_EGRID_CUH
