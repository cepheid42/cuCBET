#ifndef CUCBET_EGRID_CUH
#define CUCBET_EGRID_CUH

#include "vec2.cuh"

class Egrid {
public:
	Egrid() {
		eden = new double*[nx];
		for (int i = 0; i < nx; ++i) {
			eden[i] = new double[nz];
		}

		edep = new double*[nx + 2];
		for (int i = 0; i < nx + 2; ++i) {
			edep[i] = new double[nz + 2];
		}

		d_eden_dx = new double*[nx];
		for (int i = 0; i < nx; ++i) {
			d_eden_dx[i] = new double[nz];
		}

		d_eden_dz = new double*[nx];
		for (int i = 0; i < nx; ++i) {
			d_eden_dz[i] = new double[nz];
		}
	}

public:
	double **eden;   // shape: (nx, nz)
	double **edep;   // shape: (nx + 2, nz + 2)
	double **d_eden_dx;
	double **d_eden_dz;
};

void init_eden(Egrid& e, double ncrit) {
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < nz; ++j) {
			double val = (0.2 * ncrit * i) / (nx - 1) + (0.1 * ncrit);
			e.eden[i][j] = max(val, 0.0);
		}
	}
}

inline double get_x_coord(int i) {
	return xmin + (i * (xmax - xmin) / (nx - 1));
}

inline double get_z_coord(int j) {
	return zmin + (j * (zmax - zmin) / (nz - 1));
}

void init_eden_derivs(Egrid& e) {
	for (int i = 0; i < nx - 1; ++i) {
		for (int j = 0; j < nz - 1; ++j) {
			double x = (e.eden[i + 1][j] - e.eden[i][j]) / (get_x_coord(i + 1) - get_x_coord(i));
			e.d_eden_dx[i][j] = x;

			double z = (e.eden[i][j + 1] - e.eden[i][j]) / (get_z_coord(j + 1) - get_z_coord(j));
			e.d_eden_dz[i][j] = z;
		}
	}

	for (int i = 0; i < nz - 1; ++i) {
		e.d_eden_dx[nx - 1][i] = e.d_eden_dx[nx - 2][i];
	}

	for (int i = 0; i < nx - 1; ++i) {
		e.d_eden_dz[i][nz - 1] = e.d_eden_dz[i][nz - 2];
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
