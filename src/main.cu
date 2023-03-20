#include "Utilities/Utilities.cuh"
#include "Interpolation/Gradients.cuh"
#include "Beams/Beam2D.cuh"
#include "Utilities/Timing.cuh"

using devBeam = matrix_base<Beam2D<FPTYPE, cuda_managed>, 1, cuda_managed>;

void output_beam(Parameters& params, devBeam& beams) {
  for (uint32_t b = 0; b < beams.dims[0]; b++) {
    std::ofstream file("./outputs/beam_" + std::to_string(beams(b).ID) + ".csv");
    file << std::fixed << std::setprecision(std::numeric_limits<FPTYPE>::max_digits10);
    for (uint32_t r = 0; r < beams(b).params.nrays; ++r) {
      auto ray = beams(b).rays[r];
      file << ray.controls[0] << ", " << ray.controls[1] << ", " << ray.controls[2] << ", " << ray.controls[3] << '\n';
    }
  }
}


int main() {
  cpuTimer host_timer;
  host_timer.start();

  cudaTimer dev_timer;

  uint32_t nx = 400;
  uint32_t ny = 400;
//  uint32_t nx = 20;
//  uint32_t ny = 20;

  vec2<FPTYPE> xy_min{-5.0E-6, -5.0E-6}; // xmin and ymin in m
  vec2<FPTYPE> xy_max{5.0E-6, 5.0E-6}; // xmax and ymax in m

  auto dx = (xy_max[0] - xy_min[0]) / static_cast<FPTYPE>(nx - 1);
  auto dy = (xy_max[1] - xy_min[0]) / static_cast<FPTYPE>(ny - 1);

  auto cfl = 1.0 / sqrt(2.0);

  auto nt = static_cast<uint32_t>(FPTYPE(2 * nx) / cfl);
  auto dt = (cfl * dx) / Constants::C0;

  FPTYPE lambda = 3.51E-7; // meters
  auto beam_omega = 2.0 * Constants::PI * (Constants::C0 / lambda); // 5.366E+15 s^-1

  auto ncrit = (SQR(beam_omega) * Constants::Me * Constants::EPS0) / SQR(Constants::qe); // 9.047E+27 m^-3

  vec2<FPTYPE> beam0_norm{1.0, 0.0};
  vec2<FPTYPE> beam1_norm{0.0, 1.0};

  auto beam_radius = 2.0E-6; // meters
  auto beam_sigma = 1.7E-6;  // meters
  auto beam_intensity = 1.0E15;
  auto beam_nrays = static_cast<uint32_t>((2.0 * beam_radius) / dx);

  Parameters params{xy_min, xy_max, cfl, dx, dy, dt, ncrit, nx, ny, nt};
  BeamParams bp0{beam0_norm, beam_radius, beam_sigma, beam_intensity, beam_omega, beam_nrays};
  BeamParams bp1{beam1_norm, beam_radius, beam_sigma, beam_intensity, beam_omega, beam_nrays};

  auto beams = new devBeam(2);
  (*beams)(0) = {bp0, 0};
  (*beams)(1) = {bp1, 1};

  // Permitivitty epsilon = 1 - ne/nc
  // index of refraction = sqrt(epsilon)
  auto eps = new devMatrix<2>(nx, ny);
//  linear_permittivity_x(*eps, 0.5, 0.999);
  bilin_permittivity(*eps, 0.1, 1.0, dx, dy);
//  radial_permittivity(*eps, 0.1, 1.0, dx, dy);

//  for (uint32_t i = 0; i < nx; ++i) {
//    std::cout << (*eps)(i, 99) << " ";
//  }
//  std::cout << std::endl;

  auto eps_grad = new devVector<2>(nx, ny);
  gradient2D(*eps_grad, *eps, dx, dy);

  dev_timer.start();

  launch_rays<<<2, beam_nrays>>>(params, *beams, *eps, *eps_grad);
  cudaChk(cudaDeviceSynchronize())

  dev_timer.stop();
  auto gpu_time = dev_timer.elapsed() / 1000.0;

  output_beam(params, *beams);

  delete eps;
  delete eps_grad;
  delete beams;

  host_timer.stop();
  auto total_time = host_timer.elapsed() / 1000.0;

  std::cout << "Beams/Rays:    " << 2 * beam_nrays << " (" << 2 << "/" << beam_nrays << ")" << std::endl;
  std::cout << "GPU runtime:   " << gpu_time << std::endl;
  std::cout << "Total runtime: " << total_time << std::endl;

  return 0;
}
