#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "Rays.cuh"

inline constexpr uint32_t num_rings = 5;

float calc_intensity(float I0, float r, float w) {
  return I0 * std::exp(-2.0 * SQR(r / w));
}

struct Beam {
  uint32_t ID;
  vec3 b_norm;
  float radius;
  float intensity;
  uint32_t nRays;
  Ray* rays;

  Beam(const Parameters& params, uint32_t ID, vec3 b_norm, float r, float I0, uint32_t nrays)
  : ID(ID), b_norm(b_norm), radius(r), intensity(I0), nRays(nrays)
  {
    // Initialize rays
    cudaChk(cudaMallocManaged(&rays, nrays * sizeof(Ray)))
    cudaChk(cudaDeviceSynchronize())

    init_cylindrical_rays();    
  }

  ~Beam() {
    cudaChk(cudaDeviceSynchronize())
    cudaChk(cudaFree(rays))
  }

  void init_rays();
};

void Beam::init_rays() {

}

#endif //CBET_BEAMS_CUH
