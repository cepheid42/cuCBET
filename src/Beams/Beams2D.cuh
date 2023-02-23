#ifndef CUCBET_BEAMS2D_CUH
#define CUCBET_BEAMS2D_CUH

#include "../Utilities/Utilities.cuh"
#include "Rays2D.cuh"

struct Beam {
  uint32_t ID;
  float b_norm[2];
  float b_dist;
  float b_radius;
  float b_sigma;
  float I0;
  int nRays;
  Ray2D* rays;

  Beam(uint32_t, vec2<float>, float, float, float, float);
  ~Beam();
  void init_rays();
};

Beam::Beam(uint32_t ID, Vector2<float> _b_norm, float dist, float r, float sigma, float I0)
: ID(ID), b_norm(_b_norm), b_dist(dist), b_radius(r), b_sigma(sigma), I0(I0), nRays(128)
{
  // Initialize rays
  cudaChk(cudaMallocManaged(&rays, nRays * sizeof(Ray2D)))
  cudaChk(cudaDeviceSynchronize())

  init_rays();
}

Beam::~Beam() {
  cudaChk(cudaDeviceSynchronize())
  cudaChk(cudaFree(rays))
}

void Beam::init_beam() {
  linspace<float, nRays> phase_x;
}

#endif //CUCBET_BEAMS2D_CUH
