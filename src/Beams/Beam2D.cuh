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
__global__ void calc_grad_ne(devMatrix<2>& ne_grad, const devMatrix<2>& ne) {
  
  /*
  * todo: gradient of electron density in x, y directions
  */

}

template<typename T>
hd void interp2D(...) {

  /*
  * todo: interpolate between 4 points
  * https://en.wikipedia.org/wiki/Bilinear_interpolation~
  */

}

template<typename T>
__global__ void launch_rays(const Parameters& params, const Beam2D<T, cuda_managed>& beams, const devMatrix<2>& ne_grad) {
  auto beam_id = blockIdx.x;
  auto ray_id = threadIdx.x;

  auto beam = beams[beam_id];
  auto bparams = beam.params;

  auto delta = (2.0 * bparams.radius) / (bparams.nrays - 1);

  auto radius = -bparams.radius + (ray_id * delta);
  auto init_intensity = calculate_intensity<T>(radius, bparams.intensity, bparams.sigma);

  // Initial K-vector
  vec2<T> kvec{bparams.b_norm[0], bparams.b_norm[1]};

  // Beam 0 is x=0, Beam 1 is y=0
  ray_start[beam_id] += radius;

  // Initial position
  vec2<T> position{ray_start};

  // Intermediate points for bezier construction
  vec2<T> onethird{};
  vec2<T> twothird{};
  auto t13 = floor((1 / 3) * params.nt);
  auto t23 = floor((2 / 3) * params.nt);

  // Coefficients for various things
  auto coef1 = -params.omega / (2.0 * params.n_crit);
  auto coef2 = SQR(C) / params.omega;
  
  // Time step ray through domain
  for (uint32_t t = 0; t < params.nt; ++t) {
    // Calculate nearest nodes to ray position
    auto xi = get_node_index(ray_start[0], params.x[0], params.dx);
    auto yi = get_node_index(ray_start[1], params.y[0], params.dy);

    // Calculate k + dk
    // ne_grad should be (dne_x, dne_y)
    auto dk = coef1 * params.dt * ne_grad(xi, yi);
    kvec += dk;

    // Calculate x + dx
    auto dx = coef2 * kvec;
    position += dx;

    // Save intermediate points at t = 1/3 and t = 2/3
    if (t == t13) {
      onethird = position;
    }
    if (t == t23) {
      twothird = position;
    }
  }

  // Construct bezier curve
  auto P1 = (1.0 / 6.0) * (18.0 * onethird - 9.0 * twothird - 5.0 * ray_start + 2.0 * position);
  auto P2 = (1.0 / 6.0) * (-9.0 * onethird + 18.0 * twothird + 2.0 * ray_start - 5.0 * position);

  beam.rays[ray_id] = {ray_start, P1, P2, position, intensity};
}

#endif //CUCBET_BEAM2D_CUH
