#ifndef CUCBET_CALCULATIONS_CUH
#define CUCBET_CALCULATIONS_CUH

#include "Beam.cuh"

__global__ void calc_gain(Ray& r1, Ray& r2, int r2_num, Egrid& eg, float const1, float cs, float n_crit, const float* b1_edep) {
	unsigned tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned stride = blockDim.x * gridDim.x;


	// Pragma Unroll this?
	for (auto i = tx; i < r1.int_index; i += stride) {
		if (r1.intersections[i].r2_num != r2_num) {
			continue;
		}
		auto i1 = r1.intersections[i].r1_ind;
		auto i2 = r1.intersections[i].r2_ind;

		auto ix1 = get_x_index(r1.path[i1].x, xmax, xmin, nx);
		auto iy1 = get_y_index(r1.path[i1].y, ymax, ymin, ny);

		auto ix2 = get_x_index(r2.path[i2].x, xmax, xmin, nx);
		auto iy2 = get_y_index(r2.path[i2].y, ymax, ymin, ny);

		if (ix1 == ix2 && iy1 == iy2) {
			auto dk1 = r1.path[i1 + 1] - r1.path[i1];
			auto dk2 = r2.path[i2 + 1] - r2.path[i2];

			auto ne = eg.eden[iy1 * nx + ix1];
			auto epsilon = sqrtf(1.0f - ne / n_crit);

			auto kmag = (omega / c) * epsilon;

			auto k1 = kmag * unit_vector(dk1);
			auto k2 = kmag * unit_vector(dk2);

//			auto kiaw = std::sqrt(std::pow(k2.x() - k1.x(), 2.0f) + std::pow(k2.y() - k1.y(), 2.0f));
			auto ws = (k2 - k1).length() * cs;

			auto machnum = (((-0.4f) - (-2.4f)) / (xmax - xmin)) * (get_x_val(ix1, xmax, xmin, nx) - xmin) + (-2.4f);
			auto u_flow = fmaxf(machnum, 0.0f) * cs;
			auto eta = ((omega - omega) - (k2.x - k1.x) * u_flow) / (ws + 1.0e-10f);   // omega is not changed in this code

			auto efield1 = sqrtf(8.0f * pi * 1.0e7f * b1_edep[iy1 * nx + ix1] / c);
			auto P = (powf(iaw, 2.0f) * eta) / (powf(powf(eta, 2.0f) - 1.0f, 2.0f) + powf(iaw, 2.0f) * powf(eta, 2.0f));
			auto gain2 = const1 * powf(efield1, 2.0f) * (ne / n_crit) * (1.0f / iaw) * P;

			// Update W1_new
			eg.W_new[iy1 * nx + ix1].x = eg.W[iy1 * nx + ix1].x * expf( 1 * eg.W[iy1 * nx + ix1].y * dk1.length() * gain2 / epsilon);
			// Update W2_new
			eg.W_new[iy1 * nx + ix1].y = eg.W[iy1 * nx + ix1].y * expf(-1 * eg.W[iy1 * nx + ix1].x * dk2.length() * gain2 / epsilon);
		}
	}
}

void gpu_calc_gain(Beam& b1, Beam& b2, Egrid& eg) {
	int blockSize = 1024;
	int numBlocks = (nrays + blockSize - 1) / blockSize;

	const float const1 = std::pow(estat, 2.0f) / (4.0f * (1.0e3f * m_e) * c * omega * kb * Te * (1.0f + 3.0f * Ti / (Z * Te)));
	const float cs = 100.0f * std::sqrt(e_c * (Z * Te_eV + 3.0f * Ti_eV) / mi_kg);
	const float ncrit = 1e-6f * (std::pow(omega, 2.0f) * m_e * e_0 / std::pow(e_c, 2.0f));
	// Make this a kernel launching kernels?
	for (int r = 0; r < nrays; r++) {
		for (int p = 0; p < nrays; p++) {
			calc_gain<<<numBlocks, blockSize>>>(b1.rays[r], b2.rays[p], p, eg, const1, cs, ncrit, b1.edep);
			checkErr(cudaDeviceSynchronize());
		}
	}
}

#endif //CUCBET_CALCULATIONS_CUH
