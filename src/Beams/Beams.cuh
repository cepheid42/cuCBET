#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "../Interpolation/Interpolator.cuh"
#include "Rays.cuh"

inline constexpr uint32_t num_rings = 10;
inline constexpr uint32_t m = (num_rings + 1) * num_rings;

struct Beam {
  uint32_t ID;
  vec3 k_vec;
  float3 origin;
  float radius;
  float intensity;
  uint32_t nRays;
  Ray* rays;

  Beam(const Parameters& params, uint32_t ID, float3 origin, vec3 kvec, float r, float I, uint32_t nrays)
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

    auto rad = r / num_rings;
    auto nraysf = static_cast<float>(nrays);

    auto x0 = static_cast<uint32_t>((origin.x - params.minX() - radius) / params.dx());
    auto y0 = static_cast<uint32_t>((origin.y - params.minY() - radius) / params.dy());
    auto z0 = static_cast<uint32_t>((origin.z - params.minZ() - radius) / params.dz());

    auto x1 = static_cast<uint32_t>((origin.x - params.minX() + radius) / params.dx());
    auto y1 = static_cast<uint32_t>((origin.y - params.minY() + radius) / params.dy());
    auto z1 = static_cast<uint32_t>((origin.z - params.minZ() + radius) / params.dz());

    for (auto ring = 1; ring <= num_rings; ring++) {
      auto fring = static_cast<float>(ring);
      auto num_rays = static_cast<uint32_t>(std::round((2.0f * nraysf * fring) / m));
      auto dr2 = SQR(fring * rad);

      for (auto ray = 0; ray < num_rays; ray++) {
        for (auto i = x0; i < x1; i++) {
          for (auto j = y0; j < y1; j++) {
            auto x = (static_cast<float>(i) * params.dx());
            auto y = (static_cast<float>(j) * params.dy());

            if (SQR(x) + SQR(y) <= dr2) {
              // create ray
            }
          }
        }
      }
    }
  }

  ~Beam() {
    cudaChk(cudaDeviceSynchronize())
    cudaChk(cudaFree(rays))
  }
};

#endif //CBET_BEAMS_CUH
