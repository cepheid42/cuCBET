#ifndef CUCBET_CBET_CALCULATIONS_CUH
#define CUCBET_CBET_CALCULATIONS_CUH

#include "utilities.cuh"
#include "beam2d.cuh"

// struct intersection_record {
//   vec2<FPTYPE> point;
//   uint32_t pump_id;
//   uint32_t seed_id;
// };

// __global__ void compute_intersections(devMatrix<2>& intersections, devMatrix<1>& beams) {

// }

__global__ void cbet_gain(devMatrix<2>& epsilon,
                          matrix_base<Beam2D<FPTYPE, cuda_managed>, 1, cuda_managed>& beams) 
{
  auto eps = interp2D(epsilon, point, xymin, dx, dy);
  auto nu_eic = ...;
  // eps = 1 - ne/nc --> ne/nc = 1 - eps
  // W1(s + ds) = W1(s) * exp((-kappa_ei + sum(gamma_1j * beta_j * Wj) / sqrt(eps_eff)) * ds)
  auto kappa_ei = 2.0 * nu_eic * SQR(1 - eps) / (constants::C0 * sqrt(1 - eps));
  auto beta_j = ;
  auto gamma_1j = ;
  // 1 / sqrt(eps_eff) = min(1/sqrt(eps), 2 * sqrt(L/ds))
  auto eps_eff = ;
}



#endif //CUCBET_CBET_CALCULATIONS_CUH
