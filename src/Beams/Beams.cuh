#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "Rays.cuh"

inline constexpr uint32_t num_rings = 5;
inline constexpr float n0 = 8.32;

float calc_intensity(float I0, float r, float w) {
  return I0 * exp(-2.0 * SQR(r / w));
}

struct Beam {
  uint32_t ID;
  vec3 b_norm;
  float b_dist;
  float b_radius;
  float b_sigma;
  float I0;
  uint32_t nRays;
  Ray* rays;

  Beam(uint32_t ID, vec3 b_norm, float dist, float r, float sigma, float I0)
  : ID(ID), b_norm(b_norm), b_dist(dist), b_radius(r), b_sigma(sigma), I0(I0), nRays(128)
  {
    // Initialize rays
    cudaChk(cudaMallocManaged(&rays, nRays * sizeof(Ray<float>)))
    cudaChk(cudaDeviceSynchronize())

    init_rays();    
  }

  ~Beam() {
    cudaChk(cudaDeviceSynchronize())
    cudaChk(cudaFree(rays))
  }

  void init_rays();
};

void Beam::init_rays() {
  const auto dr = b_dist / num_rings;
  const auto a0 = std::sin(Constants::PI / n0);
  const auto beam_loc = b_dist * b_norm;

  Vector3<float> e1;

  if (b_norm[2] != 0.0) {
    e1 = unit_vector(Vector3<float>(0.0, dr, -dr * (b_norm[1] / b_norm[2])));
  } else {
    e1 = {0.0, 0.0, 1.0};
  }

  auto e2 = unit_vector(cross(e1, b_norm));

  rays[0] = Ray{b_dist * b_norm, Vector3<float>(), -b_dist * b_norm};
  int raycount = 1;

  for (auto i = 1; i <= num_rings; i++) {
    auto r = i * dr;
    auto n = std::rintf(Constants::PI / std::asin(a0 / static_cast<float>(i)));

    for (auto j = 0; j < static_cast<int>(n); j++) {
      auto theta = j * (2.0 * Constants::PI) / n;

      auto center = r * (std::cos(theta) * e1 + std::sin(theta) * e1);
      auto origin = center + beam_loc;
      auto end = center - beam_loc;

      auto intensity = calc_intensity(I0, r, b_sigma);

      rays[raycount] = Ray{origin, center, end, intensity};
      raycount++;
    }
  }

  assert(raycount == nRays);
}

void beam_to_csv(Beam& b, std::string& filename) {
  std::ofstream file("./outputs/" + filename + "_rays.csv");

  auto nt = 10;
  auto dt = 1.0 / static_cast<float>(nt);

  for (auto i = 0; i < b.nRays; i++) {
    for (auto j = 0; j < nt; j++) {
      auto t = j * dt;
      
      auto p = eval(b[i], t);
      file << p;

      if (j < nt - 1) {
        file << ", ";
      }
    }
    file << '\n'l
  }
}

#endif //CBET_BEAMS_CUH
