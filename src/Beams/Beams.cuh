#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "Rays.cuh"

inline constexpr uint32_t num_rings = 10;
inline constexpr float m = static_cast<float>((num_rings + 1) * num_rings);


float calc_intensity(float I0, float r, float w) {
  return I0 * std::exp(-2.0f * SQR(r / w));
}

struct Beam {
  uint32_t ID;
  vec3 b_norm;
  float radius;
  float intensity;
  float z_R;        // Rayleigh length
  uint32_t nRays;
  Ray* rays;

  Beam(const Parameters& params, uint32_t ID, vec3 b_norm, float r, float I0, float zR, uint32_t nrays)
  : ID(ID), b_norm(b_norm), radius(r), intensity(I0), z_R(zR), nRays(nrays)
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

  void init_cylindrical_rays();
};

void Beam::init_cylindrical_rays() {
  auto nraysf = static_cast<float>(nRays);

  auto dr = radius / num_rings;
  auto ds = m * Constants::PI * dr / nraysf;
  auto b_cyl = cartesian_to_cylindrical(b_norm);

  auto sigma = 1.7E-4f;
  auto raycount = 0;

  for (auto i = 1; i <= num_rings; i++) {
    auto ring = static_cast<float>(i);

    auto num_rays = static_cast<uint32_t>(std::round((2.0f * nraysf * ring) / m));

    auto R = ring * dr;
    auto I_ring = calc_intensity(intensity, R, sigma);

    for (auto j = 0; j < num_rays; j++) {
      auto arc = static_cast<float>(j);
      auto theta = arc * ds;

      auto ray_norm = unit_vector(
                      cylindrical_to_cartesian(
                        {b_cyl[0] + R, 
                        b_cyl[1] + theta, 
                        b_cyl[z] + z_R}
                      )
                      );
=
      rays[raycount] = Ray{ray_norm, I_ring};
      raycount++;
    }
  }
}

#endif //CBET_BEAMS_CUH
