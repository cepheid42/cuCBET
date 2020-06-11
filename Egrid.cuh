#ifndef CUCBET_EGRID_CUH
#define CUCBET_EGRID_CUH

#include "vec2.cuh"

class Egrid: public Managed {
public:
	Egrid() {
		checkErr(cudaMallocManaged(&eden,   nx * ny * sizeof(float)));
		checkErr(cudaMallocManaged(&d_eden, nx * ny * sizeof(Point)))
		checkErr(cudaMallocManaged(&W,      nx * ny * sizeof(Point)))
		checkErr(cudaMallocManaged(&W_new,  nx * ny * sizeof(Point)))
		checkErr(cudaDeviceSynchronize());
	}

	~Egrid() {
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

void init_eden_derivs(Egrid& eg) {
	// This function may possible be parallelizable...
	for (int y = 0; y < ny - 1; y++) {
		for (int x = 0; x < nx - 1; x++) {
			int index = y * nx + x;
			int x_ind = y * nx + (x + 1);
			int y_ind = (y + 1) * nx + x;

			eg.d_eden[index][0] = (eg.eden[x_ind] - eg.eden[index]) / (get_x_val(x + 1, xmax, xmin, nx) - get_x_val(x, xmax, xmin, nx));
			eg.d_eden[index][1] = (eg.eden[y_ind] - eg.eden[index]) / (get_y_val(y + 1, ymax, ymin, ny) - get_y_val(y, ymax, ymin, ny));
		}
	}

	// set last column equal to previous column
	for (int y = 0; y < ny; y++) {
		int last = y * nx + (nx - 1);
		int sec = y * nx + ( nx - 2);
		eg.d_eden[last] = eg.d_eden[sec];
	}

	// set last row equal to previous row
	for (int x = 0; x < nx; x++) {
		int last = (ny - 1) * nx + x;
		int sec = (ny - 2) * nx + x;
		eg.d_eden[last] = eg.d_eden[sec];
	}
}

void init_egrid(Egrid& eg) {
	for (int y = 0; y < ny; y++) {
		for (int x = 0; x < nx; x++) {
			int index = y * nx + x;

			float density = ((0.3f * ncrit - 0.1f * ncrit) / (xmax - xmin)) * (get_x_val(x, xmax, xmin, nx) - xmin) + (0.1f * ncrit);
			eg.eden[index] = std::max(density, 0.0f);

			float w = std::sqrt(1.0f - eg.eden[index] / ncrit); // / float(rays_per_zone);
			eg.W[index] = Point(w, w);
			eg.W_new[index] = Point(w, w);
		}
	}
	init_eden_derivs(eg);
}

// Utility Functions


#endif //CUCBET_EGRID_CUH
