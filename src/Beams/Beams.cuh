#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "Rays.cuh"

inline constexpr int num_rings = 5;
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
  int nRays;
  Ray* rays;

  Beam(uint32_t ID, vec3 b_norm, float dist, float r, float sigma, float I0)
  : ID(ID), b_norm(b_norm), b_dist(dist), b_radius(r), b_sigma(sigma), I0(I0), nRays(128)
  {
    // Initialize rays
    cudaChk(cudaMallocManaged(&rays, nRays * sizeof(Ray)))
    cudaChk(cudaDeviceSynchronize())

    init_rays();    
  }

  ~Beam() noexcept(false) {
    cudaChk(cudaDeviceSynchronize())
    cudaChk(cudaFree(rays))
  }

  void init_rays();
};

void Beam::init_rays() {
  const float dr = b_radius / num_rings;
  const float a0 = std::sin(Constants::PI / n0);
  const auto beam_loc = b_dist * b_norm;

  Vector3<float> e1;

  if (b_norm[2] != 0.0) {
    e1 = unit_vector(Vector3<float>(0.0, dr, -dr * (b_norm[1] / b_norm[2])));
  } else {
    e1 = {0.0, 0.0, 1.0};
  }

  auto e2 = unit_vector(cross(e1, b_norm));

  rays[0] = Ray{b_dist * b_norm, Vector3<float>{}, -b_dist * b_norm, I0};
  int raycount = 1;

  for (auto i = 1; i <= num_rings; i++) {
    auto r = static_cast<float>(i) * dr;
    auto n = std::rint(Constants::PI / std::asin(a0 / static_cast<float>(i)));

    for (auto j = 0; j < static_cast<int>(n); j++) {
      float theta = j * (2.0 * Constants::PI) / n;

      auto center = r * (std::cos(theta) * e1 + std::sin(theta) * e2);
      auto origin = center + beam_loc;
      auto end = center - beam_loc;

      auto intensity = calc_intensity(I0, r, b_sigma);

      rays[raycount] = Ray{origin, center, end, intensity};
      raycount++;
    }
  }

  assert(raycount == nRays);
}

template<typename T = float>
std::vector<Vector3<T>> load_beam_normals(const std::string& filename) {
  std::vector<Vector3<T>> beam_normals;

  std::ifstream file(filename);
  std::string line;

  while(std::getline(file, line)) {
    Vector3<T> bnorm;
    std::stringstream lineStream(line);

    for (auto i = 0; i < 3; i++) {
      lineStream >> bnorm[i];
    }
    beam_normals.push_back(bnorm);
  }
  
  return beam_normals;
}

// void beam_to_csv(Beam& b, const std::string& filename, int cutoff = 0) {
//   std::ofstream file(filename);

//   cutoff = std::max(cutoff, b.nRays);

//   auto nt = 4;
//   auto dt = 1.0 / static_cast<float>(nt - 1);

//   for (auto i = 0; i < cutoff; i++) {      
//     Vector3<float> p1 = b.rays[i].eval(0.0);
//     Vector3<float> p2 = b.rays[i].eval(0.5);
//     Vector3<float> p3 = b.rays[i].eval(1.0);
//     file << p1 << '\n' << p2 << '\n' << p3 << '\n';
//   }
// }

#endif //CBET_BEAMS_CUH
