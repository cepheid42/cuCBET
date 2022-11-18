#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "../Interpolation/Interpolator.cuh"
#include "Rays.cuh"

inline constexpr uint32_t num_rings = 10;
inline constexpr uint32_t m = (num_rings + 1) * num_rings;

float calc_intensity(float r, float I0, float w) {
  return I0 * std::exp(-2.0f * SQR(r / w));
}

struct Beam {
  uint32_t ID;
  vec3 k_vec;
  float3 origin;
  float radius;
  float intensity;
  uint32_t nRays;
  Ray* rays;

  Beam(const Parameters& params, 
       uint32_t ID, 
       float3 origin, 
       vec3 kvec, 
       float r, 
       float I0, 
       uint32_t nrays)
  : ID(ID),
    origin(origin),
    k_vec(kvec),
    radius(r),
    intensity(I),
    nRays(nrays)
  {
    // Initialize rays
    cudaChk(cudaMallocManaged(&rays, nrays * sizeof(Ray)))
    cudaChk(cudaDeviceSynchronize())

    auto rad = radius / num_rings;
    auto nraysf = static_cast<float>(nrays);

    auto width = 0.6f * 2.0f * radius;
    auto raycount = 0;
    nvcc
    for (auto ring = 1; ring <= num_rings; ring++) {
      auto fring = static_cast<float>(ring);
      auto num_rays = static_cast<uint32_t>(std::round((2.0f * nraysf * fring) / m));
      auto dr2 = SQR(fring * rad);

      auto I_ray = calc_intensity(rad, intensity, width);

      for (auto theta = 0; theta < num_rays; theta++) {
        rays[raycount] = {dim3(0, 0, 0), vec3(0, 0, 0), I_ray};
        raycount++;
      }
    }
  }

  ~Beam() {
    cudaChk(cudaDeviceSynchronize())
    cudaChk(cudaFree(rays))
  }
};

#endif //CBET_BEAMS_CUH
