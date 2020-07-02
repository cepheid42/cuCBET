#ifndef CUCBET_INTERSECTIONS_CUH
#define CUCBET_INTERSECTIONS_CUH

#include "Beam.cuh"

//__global__ void reduce(float* storage) {
//	const auto tid = threadIdx.x;
//
//	auto step_size = 1;
//	auto num_threads = blockDim.x;
//
//	while (num_threads > 0) {
//		if (tid < num_threads) {
//			// locations
//			auto fst = tid * step_size * 2;
//			auto snd = fst + step_size;
//
//			// Want the U value closest to 0.5.
//			float a = fabsf(storage[fst] - 0.5f);
//			float b = fabsf(storage[snd] - 0.5f);
//
//			storage[fst] = (a <= b) ? storage[fst] : storage[snd];
//		}
//
//		step_size <<= 1;
//		num_threads >>= 1;
//	}
//}


__global__ void find_intersections(Ray& r1, Ray& r2, int r2_num) {
	auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	auto stride = blockDim.x * gridDim.x;

	Point p3 = r1.path[0];
	Point p4 = r1.path[r1.endex - 1];

	for (auto t2 = tid; t2 < r2.endex - 1; t2 += stride) {
		Point p1 = r2.path[t2];
		Point p2 = r2.path[t2 + 1];

		auto numerator   = (p1.x - p3.x)*(p3.y - p4.y) - (p1.y - p3.y)*(p3.x - p4.x);
		auto denominator = (p1.x - p2.x)*(p3.y - p4.y) - (p1.y - p2.y)*(p3.x - p4.x);

		auto t = numerator / denominator;

		if (t > 0.0f && t < 1.0f) {
			printf("Found with tid: %d\n", tid);
			r1.intersections[r2_num] = p1 + (t * (p2 - p1));
		}
	}
}

void cpu_intersection(Ray& r1, Ray& r2, int r2_num) {
	for (int t1 = 1; t1 < r1.endex; t1++) {
		Point p3(r1.path[t1 - 1]);
		Point p4(r1.path[t1]);

		for (auto t2 = 1; t2 < r2.endex; t2++) {
			Point p1(r2.path[t2 - 1]);
			Point p2(r2.path[t2]);

			auto numerator = (p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x);
			auto denominator = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);

			auto t = numerator / denominator;

			if (t > 0.0f && t < 1.0f) {
				auto p = p1 + (t * (p2 - p1));
				r1.intersections[r2_num] = p;
				return;
			}
		}
	}
}



void get_intersections(Beam& b1, Beam& b2) {
	int blockSize = 1024;
	int numBlocks = (nrays + blockSize - 1) / blockSize;

	for (int r1_num = 0; r1_num < nrays; r1_num++) {
		for (int r2_num = 0; r2_num < nrays; r2_num++) {
//			find_intersections<<<numBlocks, blockSize>>>(b1.rays[r1_num], b2.rays[r2_num], r2_num);
//			checkErr(cudaDeviceSynchronize())
			cpu_intersection(b1.rays[r1_num], b2.rays[r2_num], r2_num);
		}
	}
}

#endif //CUCBET_INTERSECTIONS_CUH
