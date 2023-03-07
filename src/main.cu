#include "Utilities/Utilities.cuh"

#include "Beams/Beam2D.cuh"

int main() {
  uint32_t nx = 201;
  uint32_t ny = 201;

  vec2<FPTYPE> x{-5.0E-4, 5.0E-4}; // xmin and xmax in cm
  vec2<FPTYPE> y{-5.0E-4, 5.0E-4}; // ymin and ymax in cm

  auto dx = (x[1] - x[0]) / static_cast<FPTYPE>(nx - 1);
  auto dy = (y[1] - y[0]) / static_cast<FPTYPE>(ny - 1);

  auto cfl = 1.0 / sqrt(2.0);

  auto nt = static_cast<uint32_t>(FPTYPE(nx) / cfl);
  auto dt = (cfl * dx) / Constants::C0;

  FPTYPE lambda = 3.5E-5; // cm
  auto beam_omega = 2.0 * Constants::PI * (Constants::C0 / lambda);

  auto ncrit = (SQR(beam_omega) * Constants::Me * Constants::EPS0) / SQR(Constants::qe);

  vec2<FPTYPE> beam0_norm{0.0, 1.0};
  vec2<FPTYPE> beam1_norm{1.0, 0.0};
  auto beam_radius = 2.0E-4;
  auto beam_sigma = 0.0375;
  auto beam_intensity = 1.0E15;

  // I don't like this
  auto beam_nrays = static_cast<uint32_t>((2.0 * beam_radius) / dx);

  Parameters params{x, y, cfl, dx, dy, dt, ncrit, nx, ny, nt};
  BeamParams beam0{beam0_norm, beam_radius, beam_sigma, beam_intensity, beam_omega, beam_nrays};
  BeamParams beam1{beam1_norm, beam_radius, beam_sigma, beam_intensity, beam_omega, beam_nrays};

  
  auto ne_grad = new devMatrix<2>(nx, ny);

  launch_rays<<<2, beam_nrays>>>(params, beams, ne_grad);
  cudaChk(cudaDeviceSynchronize())

  return 0;
}
