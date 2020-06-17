#ifndef CUCBET_INTERSECTIONS_CUH
#define CUCBET_INTERSECTIONS_CUH

#include "Beam.cuh"

__global__ void find_intersections(Ray& r1, Ray& r2, int r1_num, int r2_num) {
	auto tx = threadIdx.x + blockIdx.x * blockDim.x;
	auto stride = blockDim.x * gridDim.x;

	// Maybe use a PRAGMA Unroll here?
	for (auto t1 = tx; t1 < r1.endex; t1 += stride) {
		auto x1 = r1.path[t1].x;
		auto x2 = r1.path[t1 + 1].x;
		auto y1 = r1.path[t1].y;
		auto y2 = r1.path[t1 + 1].y;

		//
		//
		// This definitely doesn't work
		// Just compares two small unrelated segments
		// and moves on, never finds an interersection.
		// ... fast though...
		//
		//
		for (auto t2 = tx; t2 < r2.endex; t2 += stride) {
			auto x3 = r2.path[t2].x;
			auto x4 = r2.path[t2 + 1].x;
			auto y3 = r2.path[t2].y;
			auto y4 = r2.path[t2 + 1].y;

			auto numerator   = (x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4);
			auto denominator = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4);

			if (denominator < 0.0f) {
				break;
			}

			auto t = numerator / denominator;
			if (0.0f <= t && t <= 1.0f) {
				r1.add_intersection(r1_num, t1, r2_num, t2);
				break;
			}
		}
	}
}

void get_intersections(Beam& b1, Beam& b2) {
	int blockSize = 1024;
	int numBlocks = (nrays + blockSize - 1) / blockSize;

	// Can I make this a kernel that launches kernels?
	for (int r = 0; r < nrays; r++) {
		for (int p = 0; p < nrays; p++) {
			find_intersections<<<numBlocks, blockSize>>>(b1.rays[r], b2.rays[p], r, p);
//			checkErr(cudaDeviceSynchronize())
		}
		checkErr(cudaDeviceSynchronize())
	}
}

#endif //CUCBET_INTERSECTIONS_CUH
