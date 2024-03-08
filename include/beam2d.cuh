#ifndef CUCBET_BEAM2D_CUH
#define CUCBET_BEAM2D_CUH

#include "utilities.cuh"
#include "parameters.cuh"
#include "interp2d.cuh"
#include "ray2d.cuh"

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

  Beam2D(const BeamParams& _params, uint32_t _id)
  : ID(_id), params(_params)
  {
    if constexpr (std::is_same<Manager, cpu_managed>::value) {
      rays = new Ray2D<T>[params.nrays];
    } else {
      cudaChk(cudaMallocManaged(&rays, params.nrays * sizeof(Ray2D<T>)))
      cudaChk(cudaDeviceSynchronize())
    }
  }

  Beam2D& operator=(Beam2D<T, Manager>&& rhs) noexcept {
    if (this != &rhs) {
      ID = rhs.ID;
      params = rhs.params;
      rays = rhs.rays;
      rhs.rays = nullptr;
    }
    return *this;
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


__global__ void launch_rays(const Parameters params,
                            matrix_base<Beam2D<FPTYPE, cuda_managed>, 1, cuda_managed>& beams,
                            const devMatrix<2>& eps,
                            const devVector<2>& eps_grad)
{
  auto beam_id = blockIdx.x;
  auto ray_id = threadIdx.x;

  auto bparams = beams(beam_id).params;

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
  auto roll = (beam_id + 1) % 2;
  ray_start[roll] = -bparams.radius + (ray_id * ray_delta);

  // Initial ray_end point
  vec2<FPTYPE> ray_end{ray_start};

  auto init_intensity = calculate_intensity(ray_start[beam_id], bparams.intensity, bparams.sigma);

  // Ray record keeping for computing points at end
  vec2<FPTYPE> records[28];
  uint32_t times[28];

  auto counter = 0;
  uint32_t total_timesteps = 0;
  uint32_t num_steps_per_save = params.nt / 28;

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

    // Stop when ray exits domain
    if (ray_end[0] >= params.xy_max[0] || ray_end[1] >= params.xy_max[1]) {
      total_timesteps = t;
      break;
    }

    // Store intermediate points
    if (t % num_steps_per_save == 0 && t > 0) {
      records[counter] = ray_end;
      times[counter] = t;
      counter++;
    }
  }

  // Get points at one-third and two-thirds
  auto first = counter / 3;
  auto second = 2 * first;

  auto s = FPTYPE(times[first - 1]) / FPTYPE(total_timesteps);
  auto t = FPTYPE(times[second - 1]) / FPTYPE(total_timesteps);

  auto A = records[first - 1];
  auto B = records[second - 1];

  auto oms = 1.0 - s;
  auto omt = 1.0 - t;
  auto tms = t - s;

  // Calculate first intermediate point
  auto den1 = 1.0 / (3.0 * s * t * oms * omt * tms);
  auto a1 = SQR(t) * omt * A;
  auto b1 = SQR(s) * oms * B;
  auto c1 = oms * omt * tms * (2.0 * s * t - s - t) * ray_start;
  auto d1 = SQR(s) * SQR(t) * tms * ray_end;
  auto P1 = den1 * (a1 - b1 + c1 + d1);

  // Calculate second intermediate point
  auto den2 = 1.0 / (3.0 * SQR(t) * omt);
  auto b2 = CUBE(omt) * ray_start;
  auto c2 = 3.0 * t * SQR(omt) * P1;
  auto d2 = CUBE(t) * ray_end;
  auto P2 = den2 * (B - b2 - c2 - d2);

  // Store new ray
  beams(beam_id).rays[ray_id] = {ray_start, P1, P2, ray_end, init_intensity};
}

#endif //CUCBET_BEAM2D_CUH
