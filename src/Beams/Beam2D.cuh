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


  auto dtau = Constants::C0 * params.dt;

  auto sample_size = params.nt / 10;
  auto records = new ray_record[sample_size];
  auto rec_num = 0;
  uint32_t total_t;

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

    if (t % 10 == 0) {
      records[rec_num] = {ray_end[0], ray_end[1], t};
      rec_num++;
    }

    if (ray_end[0] >= params.xy_max[0] || ray_end[1] >= params.xy_max[1]) {
      total_t = t;
      break;
    }
  }

  auto onet = rec_num / 3;
  auto twot = 2 * onet;

  auto onethird = vec2<FPTYPE>{records[onet].x, records[onet].y};
  auto twothird = vec2<FPTYPE>{records[twot].x, records[twot].y};
  
  // Construct bezier curve
  auto P1 = (1.0 / 6.0) * (18.0 * onethird - 9.0 * twothird - 5.0 * ray_start + 2.0 * ray_end);
  auto P2 = (1.0 / 6.0) * (-9.0 * onethird + 18.0 * twothird + 2.0 * ray_start - 5.0 * ray_end);

  beam.rays[ray_id] = {ray_start, P1, P2, ray_end, init_intensity};
}

#endif //CUCBET_BEAM2D_CUH
