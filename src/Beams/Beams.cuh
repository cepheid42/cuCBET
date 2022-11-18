#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "Rays.cuh"

inline constexpr uint32_t num_rings = 10;
inline constexpr float m = static_cast<float>((num_rings + 1) * num_rings);

vec3 spherical_to_cartesian(float r, float phi, float theta) {
  // Make sure I agree with Wikipedia about Theta == Theta and Phi == Phi
  auto x = r * std::sin(theta) * std::cos(phi);
  auto y = r * std::sin(theta) * std::sin(phi);
  auto z = r * std::cos(theta);

  return {x, y, z};
}

float calc_intensity(float I0, float r, float w) {
  return I0 * std::exp(-2.0f * SQR(r / w));
}

struct Beam {
  uint32_t ID;
  float3 origin;
  float radius;
  float intensity;
  float focal;
  uint32_t nRays;
  Ray* rays;

  Beam(const Parameters& params, 
       uint32_t ID, 
       float3 origin, 
       float radius,
       float I0,
       float focal_length,
       uint32_t nrays)
  : ID(ID),
    origin(origin),
    radius(radius),
    intensity(I0),
    focal(focal_length),
    nRays(nrays)
  {
    // Initialize rays
    cudaChk(cudaMallocManaged(&rays, nrays * sizeof(Ray)))
    cudaChk(cudaDeviceSynchronize())

    auto nraysf = static_cast<float>(nrays);

    auto dr = radius / num_rings;
    auto ds = m * Constants::PI * dr / nraysf;

    auto sigma = 1.7E-4f;
    auto raycount = 0;

    for (auto i = 1; i <= num_rings; i++) {
      auto ring = static_cast<float>(i);

      auto num_rays = static_cast<uint32_t>(std::round((2.0f * nraysf * ring) / m));

      auto r = ring * ds;
      auto theta = std::atan(r / focal);

      auto I_ring = calc_intensity(intensity, r, sigma);

      for (auto j = 0; j < num_rays; j++) {
        auto arc = static_cast<float>(j);
        auto phi = arc * ds;

        auto kvec = spherical_to_cartesian(r, phi, theta);

        

        rays[raycount] = Ray{dim3(0, 0, 0), vec3(0, 0, 0), I_ring};
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
