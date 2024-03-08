#include <iostream>
#include <iomanip>
#include <fstream>

#include "utilities.cuh"
#include "array.cuh"
#include "gradients.cuh"
#include "beam2d.cuh"
#include "timing.cuh"

using devBeam = matrix_base<Beam2D<fptype, cuda_managed>, 1, cuda_managed>;

void output_beam(Parameters& params, devBeam& beams) {
  for (size_t b = 0; b < beams.dims[0]; b++) {
    std::ofstream file("./outputs/beam_" + std::to_string(beams(b).ID) + ".csv");
    file << std::fixed << std::setprecision(std::numeric_limits<fptype>::max_digits10);
    for (size_t r = 0; r < beams(b).params.nrays; ++r) {
      auto ray = beams(b).rays[r];
      file << ray.controls[0] << ", " << ray.controls[1] << ", " << ray.controls[2] << ", " << ray.controls[3] << '\n';
    }
  }
}


int main() {
  Array2d<float> arr1(10, 10);

//  cpuTimer host_timer;
//  host_timer.start();
//
//  cudaTimer dev_timer;
//
//  size_t nx = 400;
//  size_t ny = 400;
////  size_t nx = 20;
////  size_t ny = 20;
//
//  vec2<fptype> xy_min{-5.0E-6, -5.0E-6}; // xmin and ymin in m
//  vec2<fptype> xy_max{5.0E-6, 5.0E-6}; // xmax and ymax in m
//
//  auto dx = (xy_max[0] - xy_min[0]) / static_cast<fptype>(nx - 1);
//  auto dy = (xy_max[1] - xy_min[0]) / static_cast<fptype>(ny - 1);
//
//  auto cfl = 1.0 / sqrt(2.0);
//
//  auto nt = static_cast<size_t>(fptype(2 * nx) / cfl);
//  auto dt = (cfl * dx) / constants::C0;
//
//  fptype lambda = 3.51E-7; // meters
//  auto beam_omega = 2.0 * constants::PI * (constants::C0 / lambda); // 5.366E+15 s^-1
//
//  auto ncrit = (math::SQR(beam_omega) * constants::Me * constants::EPS0) / math::SQR(constants::qe); // 9.047E+27 m^-3
//
//  vec2<fptype> beam0_norm{1.0, 0.0};
//  vec2<fptype> beam1_norm{0.0, 1.0};
//
//  auto beam_radius = 2.0E-6; // meters
//  auto beam_sigma = 1.7E-6;  // meters
//  auto beam_intensity = 1.0E15;
//  auto beam_nrays = static_cast<size_t>((2.0 * beam_radius) / dx);
//
//  Parameters params{xy_min, xy_max, cfl, dx, dy, dt, ncrit, nx, ny, nt};
//  BeamParams bp0{beam0_norm, beam_radius, beam_sigma, beam_intensity, beam_omega, beam_nrays};
//  BeamParams bp1{beam1_norm, beam_radius, beam_sigma, beam_intensity, beam_omega, beam_nrays};
//
//  auto beams = new devBeam(2);
//  (*beams)(0) = {bp0, 0};
//  (*beams)(1) = {bp1, 1};
//
//  // Permitivitty epsilon = 1 - ne/nc
//  // index of refraction = sqrt(epsilon)
//  auto eps = new devMatrix<2>(nx, ny);
////  linear_permittivity_x(*eps, 0.5, 0.999);
//  bilin_permittivity(*eps, 0.1, 1.0, dx, dy);
////  radial_permittivity(*eps, 0.1, 1.0, dx, dy);
//
//
//  auto eps_grad = new devVector<2>(nx, ny);
//  gradient2D(*eps_grad, *eps, dx, dy);
//
//  dev_timer.start();
//
//  launch_rays<<<2, beam_nrays>>>(params, *beams, *eps, *eps_grad);
//  cudaChk(cudaDeviceSynchronize())
//
//  dev_timer.stop();
//  auto gpu_time = dev_timer.elapsed() / 1000.0;
//
//  output_beam(params, *beams);
//
//  delete eps;
//  delete eps_grad;
//  delete beams;
//
//  host_timer.stop();
//  auto total_time = host_timer.elapsed() / 1000.0;
//
//  std::cout << "Beams/Rays:    " << 2 * beam_nrays << " (" << 2 << "/" << beam_nrays << ")" << std::endl;
//  std::cout << "GPU runtime:   " << gpu_time << std::endl;
//  std::cout << "Total runtime: " << total_time << std::endl;

  return 0;
}
