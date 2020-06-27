#ifndef CUCBET_INTERSECTIONS_CUH
#define CUCBET_INTERSECTIONS_CUH

#include "Beam.cuh"

__global__ void find_intersections(Ray& r1, Ray& r2, int r2_num) {
	auto tx = threadIdx.x + blockIdx.x * blockDim.x;
	auto stride = blockDim.x * gridDim.x;

	// Make it so that you pick every segment of Ray2, and then calculate U against all points of ray1

	// Figure out which orientation is needed
	// u or t?
	// keep r1 -> p1, p2 | r2 -> p3, p4
	// or   r1 -> p3, p4 | r2 -> p1, p2

	Point p3 = r1.path[0];
	Point p4 = r1.path[r1.endex - 1];

	for (auto t2 = tx; t2 < r2.endex - 1; t2 += stride) {
		Point p1 = r2.path[t2];
		Point p2 = r2.path[t2 + 1];

		auto numerator   = (p1.x - p2.x)*(p1.y - p3.y) - (p1.y - p2.y)*(p1.x - p3.x);
		auto denominator = (p1.x - p2.x)*(p3.y - p4.y) - (p1.y - p2.y)*(p3.x - p4.x);

		auto u = -numerator / denominator;

		if (u >= 0.0f && u <= 1.0f) {
			printf("u= %f | (%f, %f)\n", u, p1.x + u * (p2.x - p1.x), p1.y + u * (p2.y - p1.y));
			r1.intersections[r2_num].x = p3.x + u * (p4.x - p3.x);
			r1.intersections[r2_num].y = p3.y + u * (p4.y - p3.y);
		}
	}
}

void get_intersections(Beam& b1, Beam& b2) {
	int blockSize = 1024;
	int numBlocks = (nrays + blockSize - 1) / blockSize;

	for (int r2_num = 0; r2_num < nrays; r2_num++) {
		for (int r1_num = 0; r1_num < nrays; r1_num++) {
			find_intersections<<<numBlocks, blockSize>>>(b1.rays[r1_num], b2.rays[r2_num], r2_num);

			checkErr(cudaDeviceSynchronize())
		}
	}
}

#endif //CUCBET_INTERSECTIONS_CUH
