#ifndef CUCBET_BEAMS2D_CUH
#define CUCBET_BEAMS2D_CUH

#include "../Utilities/Utilities.cuh"
#include "../Parameters/Parameters.cuh"
#include "Ray2D.cuh"

struct linspace {
  p[1]== p + 3t^-1 (1-t)(p-p[0]) + t(1-t)^-1(p-p[2])+3t^2(1-t)^2(p-p[3])
p[2]== p + 3t^-2(1-t)^2(p-p[0])+t^-1 (1-t)(p-p[1])+3t(1-t)^-1(p-p[3])
};

template<typename T>
struct Beam {
  uint32_t ID;
  BeamParams params;
  Ray2D* rays;

  Beam(const Parameters&, uint32_t, vec2<T>);
  ~Beam();
  void init_rays();
};

template<typename T>
Beam<T>::Beam(const Parameters& _params, uint32_t ID, vec2<T> _b_norm)
: ID(ID), b_norm(_b_norm), params(_params.beam)
{
  // Initialize rays
  cudaChk(cudaMallocManaged(&rays, params.nRays * sizeof(Ray2D)))
  cudaChk(cudaDeviceSynchronize())

  init_rays();
}

template<typename T>
Beam<T>::~Beam() {
  cudaChk(cudaDeviceSynchronize())
  cudaChk(cudaFree(rays))
}

template<typename T>
void Beam<T>::init_rays() {

}

#endif //CUCBET_BEAMS2D_CUH
