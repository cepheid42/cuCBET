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
  Ray2D<T> *rays;

  Beam2D(const BeamParams &_params, uint32_t _id)
    : ID(_id), params(_params) {
    if constexpr (std::is_same<Manager, cpu_managed>::value) {
      rays = new Ray2D<T>[params.nrays];
    } else {
      cudaChk(cudaMallocManaged(&rays, params.nrays * sizeof(Ray2D<T>)))
      cudaChk(cudaDeviceSynchronize())
    }
  }

  ~Beam2D() {
    if constexpr (std::is_same<Manager, cpu_managed>::value) {
      delete[] rays;
    } else {
      cudaChk(cudaDeviceSynchronize())
      cudaChk(cudaFree(rays))
    }
  }
};

struct ray_record {
  FPTYPE x, y;
  uint32_t t;
};

__global__ void launch_rays(const Parameters params,
                            Beam2D<FPTYPE, cuda_managed>& beam,
                            const devMatrix<2>& eps,
                            const devVector<2>& eps_grad)
{
  auto ray_id = threadIdx.x;
  auto bparams = beam.params;

  auto dtau = Constants::C0 * params.dt;
  auto ray_delta = (2.0 * bparams.radius) / (bparams.nrays - 1);

  // Initial position and k-vector
  vec2<FPTYPE> ray_start{bparams.b_norm[0] * params.xy_min[0],
                         bparams.b_norm[1] * params.xy_min[1]};

  // Interpolate ray position into permittivity
  // kvec = k' is the normalized k vector where |k| = sqrt(eps)
  // k' = sqrt(eps(x, y)) * k_hat
  // k_hat == beam normal
  auto kmag = sqrt(interp2D(eps, ray_start, params.xy_min, params.dx, params.dy));
  vec2<FPTYPE> kvec{kmag * bparams.b_norm};

  // Beam 0 increments y-dir, Beam 1 increments x-dir
  auto roll = (beam.ID + 1) % 2;
  ray_start[roll] = -bparams.radius + (ray_id * ray_delta);

  // Initial ray_end point
  vec2<FPTYPE> ray_end{ray_start};

  auto init_intensity = calculate_intensity(ray_start[beam.ID], bparams.intensity, bparams.sigma);

  // Ray record keeping for computing points at end
  ray_record records[28];
  auto counter = 0;
  uint32_t total_count = 0;
  auto num_steps_per_save = params.nt / 28;

  // Time step ray through domain
  for (uint32_t t = 0; t < params.nt; ++t) {
    // Calculate k + dk
    // dk/dtau = 1/2 grad(eps)
    auto dk = 0.5 * dtau * interp2D(eps_grad, ray_end, params.xy_min, params.dx, params.dy);
    kvec += dk;

    // Calculate x + dx
    // dx/dtau = k'
    auto dx = dtau * kvec;
    ray_end += dx;

    if (ray_end[0] >= params.xy_max[0] || ray_end[1] >= params.xy_max[1]) {
      total_count = t;
      break;
    }

    if (t % num_steps_per_save == 0 && t != 0) {
      records[counter] = {ray_end[0], ray_end[1], t};
      counter++;
    }
  }

  auto first = counter / 3;
  auto second = (2 * counter) / 3;

  auto t_one = FPTYPE(records[first].t) / FPTYPE(total_count);
  auto t_two = FPTYPE(records[second].t) / FPTYPE(total_count);

  auto A1 = vec2<FPTYPE>{records[first].x, records[first].y};
  auto A2 = vec2<FPTYPE>{records[second].x, records[second].y};

  auto alpha1 = 3.0 * SQR(1.0 - t_one);
  auto alpha2 = 3.0 * SQR(1.0 - t_two);

  auto c1 = A1 - CUBE(1.0 - t_one) * ray_start - CUBE(t_one) * ray_end;
  auto c2 = A2 - CUBE(1.0 - t_two) * ray_start - CUBE(t_two) * ray_end;

  auto P2 = (c2 - ((alpha2 * t_two) / (alpha1 * t_one)) * c1) / (alpha2 * (SQR(t_two) - t_one * t_two));
  auto P1 = (c1 / (alpha1 * t_one)) - t_one * P2;

  beam.rays[ray_id] = {ray_start, P1, P2, ray_end, init_intensity};
}

#endif //CUCBET_BEAM2D_CUH
