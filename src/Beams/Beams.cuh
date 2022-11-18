#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "Rays.cuh"

inline constexpr uint32_t num_rings = 10;
inline constexpr float m = static_cast<float>((num_rings + 1) * num_rings);

vec3 polar_to_cartesian(float r, float phi) {

}

vec3 spherical_to_cartesian(float r, float phi, float theta) {
  auto x = r * std::sin(theta) * std::cos(phi);
  auto y = r * std::sin(theta) * std::sin(phi);
  auto z = r * std::cos(theta);

  return {x, y, z};
}

vec3 cartesian_to_spherical(float x, float y, float z) {
  auto r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  auto theta = std::atan2(std::sqrt(SQR(x) + SQR(y)), z);
  auto phi = std::atan2(y, x);

  return {r, phi, theta};
}

vec3 rotate(const vec3& v, const float theta) {
  auto cos = std::cos(theta);
  auto sin = std::sin(theta);
  auto xp = v.x() * cos + v.y() * sin;
  auto yp = -v.x() * sin + v.y() * cos;

  return {xp, yp, v.z()};
}

float calc_intensity(float I0, float r, float w) {
  return I0 * std::exp(-2.0f * SQR(r / w));
}

struct Beam {
  uint32_t ID;
  vec3 origin;
  vec3 k_vec;
  float radius;
  float intensity;
  float focal;
  uint32_t nRays;
  Ray* rays;

  Beam(const Parameters& params, 
       uint32_t ID, 
       vec3 o,
       vec3 k,
       float r,
       float I0,
       float f,
       uint32_t nrays)
  : ID(ID),
    origin(o),
    k_vec(k),
    radius(r),
    intensity(I0),
    focal(f),
    nRays(nrays)
  {
    // Initialize rays
    cudaChk(cudaMallocManaged(&rays, nrays * sizeof(Ray)))
    cudaChk(cudaDeviceSynchronize())

    auto nraysf = static_cast<float>(nrays);

    auto dr = radius / num_rings;
    auto ds = m * Constants::PI * dr / nraysf;

    auto z_norm = vec3{0.0, 0.0, 1.0};

    // Find rotation angle to align x-y to plane of origin.
    auto 

    // Find angle for rotating ray vectors to align with the k-vector.
    // The spherical coords below assume theta aligns along z-axis.
    auto k_unit = unit_vector(k_vec);
    auto k_rot_angle = std::acos(dot(k_unit, z_norm) / (k_unit.length()));

    auto sigma = 1.7E-4f;
    auto raycount = 0;

    for (auto i = 1; i <= num_rings; i++) {
      auto ring = static_cast<float>(i);

      auto num_rays = static_cast<uint32_t>(std::round((2.0f * nraysf * ring) / m));

      auto R = ring * ds;
      auto theta = std::atan2(r, focal);

      auto I_ring = calc_intensity(intensity, R, sigma);

      for (auto j = 0; j < num_rays; j++) {
        auto arc = static_cast<float>(j);
        auto phi = arc * ds;

        auto

        auto k_ray = spherical_to_cartesian(r, phi, theta);
        auto k_ray_rotated = rotate(k_ray, k_rot_angle);

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
