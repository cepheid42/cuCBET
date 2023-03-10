#ifndef CUCBET_BEAM2D_CUH
#define CUCBET_BEAM2D_CUH

#include "../Utilities/Utilities.cuh"
#include "../Parameters/Parameters.cuh"
#include "../Interpolation/Interp2D.cuh"
#include "Ray2D.cuh"

#define hd __host__ __device__


template<typename T>
hd T calculate_intensity(T radius, T I0, T sigma) {
  return I0 * exp(-2.0 * SQR(radius / sigma));
}

template<typename T, class Manager>
struct Beam2D : Manager {
  uint32_t ID;
  BeamParams params;
  Ray2D<T>* rays;

  Beam2D(const BeamParams& _params, uint32_t _id)
  : ID(_id), params(_params)
  {
    if constexpr(std::is_same<Manager, cpu_managed>::value) {
      rays = new Ray2D<T>[params.nrays];
    } else {
      cudaChk(cudaMallocManaged(&rays, params.nrays * sizeof(Ray2D<T>)))
      cudaChk(cudaDeviceSynchronize())
    }
  }

  ~Beam2D() {
    if constexpr(std::is_same<Manager, cpu_managed>::value) {
      delete[] rays;
    } else {
      cudaChk(cudaDeviceSynchronize())
      cudaChk(cudaFree(rays))
    }
  }
};

__global__ void launch_rays(const Parameters params, Beam2D<FPTYPE, cuda_managed>& beam, const devVector<2>& ne_grad) {
  auto ray_id = threadIdx.x;
  auto bparams = beam.params;

  auto delta = (2.0 * bparams.radius) / bparams.nrays;


  // Initial position and k-vector
  vec2<FPTYPE> ray_start{bparams.b_norm[0] * params.xy_min[0],
                         bparams.b_norm[1] * params.xy_min[1]};

  // Beam 0 increments y-dir, Beam 1 increments x-dir
  auto roll = (beam.ID + 1) % 2;
  ray_start[roll] = -bparams.radius + (ray_id * delta);

  auto init_intensity = calculate_intensity(ray_start[roll], bparams.intensity, bparams.sigma);

  vec2<FPTYPE> kvec{ray_start};

  // Initial ray_end point
  vec2<FPTYPE> ray_end{ray_start};

  // Intermediate points for bezier construction
  vec2<FPTYPE> onethird{};
  vec2<FPTYPE> twothird{};
  auto t13 = floor(0.333 * params.nt);
  auto t23 = floor(0.666 * params.nt);

  // Coefficients for various things
  auto coef1 = -(bparams.omega * params.dt) / 2.0;
  auto coef2 = (SQR(Constants::C0) * params.dt) / bparams.omega;

  if (ray_id == 0) {
    printf("(%E, %E)\n", kvec[0], kvec[1]);
    __syncthreads();
  }

  // Time step ray through domain
  for (uint32_t t = 0; t < params.nt; ++t) {
    // Calculate k + dk
    // interpolate gradient of ne
    auto dk = coef1 * interp2D(ne_grad, ray_end, params.xy_min, params.dx, params.dy);
    kvec += dk;

    if (ray_id == 0) {
      printf("(%E, %E) + (%E, %E)\n", kvec[0], kvec[1], dk[0], dk[1]);
      __syncthreads();
    }

    // Calculate x + dx
    auto dx = coef2 * kvec;
    ray_end += dx;
//
//    if (ray_id == 0) {
//      printf("%E, %E\n", kvec[0], kvec[1]);
//      __syncthreads();
//    }

    // Save intermediate points at t = 1/3 and t = 2/3
    if (t == t13) {
      onethird = ray_end;
    }
    if (t == t23) {
      twothird = ray_end;
    }
  }

  // Construct bezier curve
  auto P1 = (1.0 / 6.0) * (18.0 * onethird - 9.0 * twothird - 5.0 * ray_start + 2.0 * ray_end);
  auto P2 = (1.0 / 6.0) * (-9.0 * onethird + 18.0 * twothird + 2.0 * ray_start - 5.0 * ray_end);

  beam.rays[ray_id] = {ray_start, P1, P2, ray_end, init_intensity};
//  auto cur_ray = beam.rays[ray_id];
//  cur_ray.update_control(0, ray_start);
//  cur_ray.update_control(1, P1);
//  cur_ray.update_control(2, P2);
//  cur_ray.update_control(3, ray_end);
//  cur_ray.intensity = init_intensity;
}

#endif //CUCBET_BEAM2D_CUH
