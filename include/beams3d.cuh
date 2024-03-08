#ifndef CBET_BEAMS_CUH
#define CBET_BEAMS_CUH

#include "../Utilities/Utilities.cuh"
#include "Rays3D.cuh"

inline constexpr int num_rings = 5;
inline constexpr float n0 = 8.32;

float calc_intensity(float I0, float r, float w) {
  return I0 * exp(-2.0 * SQR(r / w));
}

struct Beam {

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


#endif //CBET_BEAMS_CUH
