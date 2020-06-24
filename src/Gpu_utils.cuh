#ifndef CUCBET_GPU_UTILS_CUH
#define CUCBET_GPU_UTILS_CUH

#define __hd__ __host__ __device__

// Error Checking for CUDA functions
#define checkErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif //CUCBET_GPU_UTILS_CUH
