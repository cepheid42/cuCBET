#ifndef CUCBET_BEAM2D_CUH
#define CUCBET_BEAM2D_CUH

#include "../Utilities/Utilities.cuh"
#include "../Parameters/Parameters.cuh"
#include "Ray2D.cuh"

#define hd __host__ __device__

template<typename T>
hd T calculate_intensity(T radius, T I0, T sigma) {
  return I0 * exp(-2.0 * SQR(radius / sigma));
}

template<typename T>
hd uint32_t get_node_index(T x, T xmin, T dx) {
  return uint32_t(rint(((x - xmin) / dx)));
}

template<typename T, class Manager>
struct Beam2D : Manager {
  uint32_t ID;
  BeamParams params;
  Ray2D<T>* rays;

  Beam2D(const BeamParams& _params, uint32_t _id)
  : ID(_id), params(_params)
  {
    Manager::allocate_data(rays, params.nrays);
  }

  ~Beam2D() {
    Manager::deallocate_data(rays);
  }
};

template<typename T>
__global__ void launch_rays(const Parameters& params, const Beam2D<T, cuda_managed>& beams, const devMatrix<2>& e_density) {
  auto beam_id = blockIdx.x;
  auto ray_id = threadIdx.x;

  auto beam = beams[beam_id];
  auto bparams = beam.params;

  auto delta = (2.0 * bparams.radius) / (bparams.nrays - 1);

  auto radius = -bparams.radius + (ray_id * delta);
  auto intensity = calculate_intensity<T>(radius, bparams.intensity, bparams.sigma);
  auto omega = 2.0 * Constants::PI * (Constants::C0 / bparams.lambda);

  vec2<T> ray_start{bparams.b_norm[0], bparams.b_norm[1]};
  // Beam 0 is x=0, Beam 1 is y=0
  ray_start[beam_id] += radius;

  auto xi = get_node_index(ray_start[0], params.x[0], params.dx);
  auto yi = get_node_index(ray_start[1], params.y[0], params.dy);
  auto omega_p_sqr = (e_density(xi, yi) * SQR(Constants::qe)) / (Constants::EPS0 * Constants::Me);
  auto k = sqrt(SQR(omega) - omega_p_sqr) / Constants::C0;
}


//template<typename T>
//void Beam2D<T>::init_rays() {

//}

#endif //CUCBET_BEAM2D_CUH
