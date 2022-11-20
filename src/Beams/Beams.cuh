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
  vec3 b_norm
  float radius;
  float intensity;
  float z_R;        // Rayleigh length
  uint32_t nRays;
  Ray* rays;

  Beam(const Parameters& params, 
       uint32_t ID, 
       vec3 b_norm,
       float r,
       float I0,
       float zR,
       uint32_t nrays)
  : ID(ID),
    b_norm(b_norm),
    radius(r),
    intensity(I0),
    z_R(zR),
    nRays(nrays)
  {
    // Initialize rays
    cudaChk(cudaMallocManaged(&rays, nrays * sizeof(Ray)))
    cudaChk(cudaDeviceSynchronize())

    auto nraysf = static_cast<float>(nrays);

    /*
    * Each beam is composed of concentric rings of rays at equal distance, 
    *         dr = (beam radius) / (number of rings).
    * The distance between points in the same ring are equally spaced
    * by the circumference of the ring and the total number of rays,
    *         ds = (2*pi*dr) / (num rays). The number
    * of rays per rings is 
    *         (2 * num rays * ring num) / (num rings * (num rings + 1))
    * The beam intensity is a function of the radius. The angle phi is the
    * position around the vertical axis. The normal vector of the ray is
    * calculated by converting the cylindrical coordinates to cartesian
    * and made into a unit vector.
    *
    * I'm not entirely convinced the rays are pointing along the beam normal 
    */
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

        auto ray_norm = unit_vector(cylindrical_to_cartesian({b_cyl[0] + R, b_cyl[1] + theta, b_cyl[z]}));
=
        rays[raycount] = Ray{ray_norm, I_ring};
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
