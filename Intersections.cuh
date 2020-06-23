#ifndef CUCBET_INTERSECTIONS_CUH
#define CUCBET_INTERSECTIONS_CUH

#include "Beam.cuh"

__global__ void sub_inter(float x1, float y1, float x2, float y2, Ray& r1, Ray& r2, int r2_num) {
	auto tx = threadIdx.x + blockIdx.x * blockDim.x;
	auto stride = blockDim.x * gridDim.x;

	for (auto t2 = tx; t2 < r2.endex; t2 += stride) {
		auto x3 = r2.path[t2].x;
		auto y3 = r2.path[t2].y;

		auto x4 = r2.path[t2 + 1].x;
		auto y4 = r2.path[t2 + 1].y;

		auto numerator   = (x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4);
		auto denominator = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4);

		if (denominator == 0.0f) {
			break;
		}

		auto t = numerator / denominator;

		if (0.0f <= t && t <= 1.0f) {
			r1.intersections[r2_num].x = x1 + t * (x2 - x1);
			r1.intersections[r2_num].y = y1 + t * (y2 - y1);
			break;
		}
	}
}

__global__ void find_intersections(Ray& r1, Ray& r2, int r2_num) {
	auto tx = threadIdx.x + blockIdx.x * blockDim.x;
	auto stride = blockDim.x * gridDim.x;

	int blockSize = 1024;
	int numBlocks = (nrays + blockSize - 1) / blockSize;

	for (auto t1 = tx; t1 < r1.endex; t1 += stride) {
		auto x1 = r1.path[t1].x;
		auto y1 = r1.path[t1].y;

		auto x2 = r1.path[t1 + 1].x;
		auto y2 = r1.path[t1 + 1].y;

		sub_inter<<<numBlocks, blockSize>>>(x1, y1, x2, y2, r1, r2, r2_num);
	}
}

void get_intersections(Beam& b1, Beam& b2) {
	int blockSize = 1024;
	int numBlocks = (nrays + blockSize - 1) / blockSize;

	// Can I make this a kernel that launches kernels?
	for (int r1_num = 0; r1_num < nrays; r1_num++) {
		for (int r2_num = 0; r2_num < nrays; r2_num++) {
			find_intersections<<<numBlocks, blockSize>>>(b1.rays[r1_num], b2.rays[r2_num], r2_num);
			checkErr(cudaDeviceSynchronize())
		}
	}
}

#endif //CUCBET_INTERSECTIONS_CUH
