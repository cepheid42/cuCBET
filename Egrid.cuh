#ifndef CUCBET_EGRID_CUH
#define CUCBET_EGRID_CUH

#include "vec2.cuh"

class Egrid: public Managed {
public:
	Egrid() = default;

	void allocate(param_struct *pm) {
		checkErr(cudaMallocManaged(&eden, pm->nx * pm->ny * sizeof(float)))
		checkErr(cudaMallocManaged(&d_eden, pm->nx * pm->ny * sizeof(Point)));
		checkErr(cudaMallocManaged(&W, pm->nx * pm->ny * sizeof(Point)));
		checkErr(cudaMallocManaged(&W_new, pm->nx * pm->ny * sizeof(Point)));
		checkErr(cudaDeviceSynchronize());
	}

	void deallocate() const {
		checkErr(cudaDeviceSynchronize());
		checkErr(cudaFree(eden));
		checkErr(cudaFree(d_eden));
		checkErr(cudaFree(W));
		checkErr(cudaFree(W_new));
	}


public:
	float* eden;
	Point* d_eden;
	Point* W;
	Point* W_new;
};

void init_eden_derivs(Egrid* eg, param_struct* pm) {
	// This function may possible be parallelizable...
	for (int y = 0; y < pm->ny - 1; y++) {
		for (int x = 0; x < pm->nx - 1; x++) {
			int index = y * pm->nx + x;
			int x_ind = y * pm->nx + (x + 1);
			int y_ind = (y + 1) * pm->nx + x;

			eg->d_eden[index][0] = (eg->eden[x_ind] - eg->eden[index]) / (get_x_val(x + 1, pm->xmax, pm->xmin, pm->nx) - get_x_val(x, pm->xmax, pm->xmin, pm->nx));
			eg->d_eden[index][1] = (eg->eden[y_ind] - eg->eden[index]) / (get_y_val(y + 1, pm->ymax, pm->ymin, pm->ny) - get_y_val(y, pm->ymax, pm->ymin, pm->ny));
		}
	}

	// set last column equal to previous column
	for (int y = 0; y < pm->ny; y++) {
		int last = y * pm->nx + (pm->nx - 1);
		int sec = y * pm->nx + (pm-> nx - 2);
		eg->d_eden[last] = eg->d_eden[sec];
	}

	// set last row equal to previous row
	for (int x = 0; x < pm->nx; x++) {
		int last = (pm->ny - 1) * pm->nx + x;
		int sec = (pm->ny - 2) * pm->nx + x;
		eg->d_eden[last] = eg->d_eden[sec];
	}
}

void init_egrid(Egrid* eg, param_struct* pm) {
	for (int y = 0; y < pm->ny; y++) {
		for (int x = 0; x < pm->nx; x++) {
			int index = y * pm->nx + x;

			float density = ((0.3f * pm->ncrit - 0.1f * pm->ncrit) / (pm->xmax - pm->xmin)) * (get_x_val(x, pm->xmax, pm->xmin, pm->nx) - pm->xmin) + (0.1f * pm->ncrit);
			eg->eden[index] = std::max(density, 0.0f);

			float w = std::sqrt(1.0f - eg->eden[index] / pm->ncrit); // / float(rays_per_zone);
			eg->W[index] = Point(w, w);
			eg->W_new[index] = Point(w, w);
		}
	}
	init_eden_derivs(eg, pm);
}

// Utility Functions
void save_egrid_to_files(Egrid* eg, param_struct* pm) {
	const std::string output_path = "./Outputs/";

	// Write eden to file
	std::ofstream eden_file(output_path + "eden.csv");
	eden_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int i = pm->ny - 1; i >= 0; i--) {
		for (int j = 0; j < pm->nx; j++) {
			int index = i * pm->nx + j;
			eden_file << eg->eden[index];
			if (j != pm->nx - 1) {
				eden_file << ", ";
			}
		}
		eden_file << "\n";
	}
	eden_file.close();
}

#endif //CUCBET_EGRID_CUH
