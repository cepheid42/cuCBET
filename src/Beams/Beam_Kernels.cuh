#ifndef CBET_BEAM_KERNELS_CUH
#define CBET_BEAM_KERNELS_CUH

#include "../Utilities/Utilities.cuh"
#include "Beams.cuh"

__global__ void compute_intersections(const Beam& b1, const Beam& b2, bool* intersects) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  // Cache ray for this thread
  Ray b1_ray = b1.rays[tid];
  
  // Load all rays from beam 2 into shared memory
  __shared__ Ray b2_rays[128];  
  for (int thd = tid; tid < 128; thd += blockDim.x) {
    b2_rays[thd] = b2.rays[thd];
  }
  __syncthreads();

  // Compute intersections between rays
  for (int thd = tid; tid < 128; thd += blockDim.x) {
    intersects[thd + 128 * tid] = rays_bb_intersect(b1_ray, b2_rays[thd]);
  }

}

#endif // CBET_BEAM_KERNELS_CUH