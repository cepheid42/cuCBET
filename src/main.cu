#include "Utilities/Utilities.cuh"
#include "Interpolation/Gradients.cuh"
#include "Beams/Beam2D.cuh"

using devBeam = Beam2D<FPTYPE, cuda_managed>;

void output_beam(Parameters& params, devBeam& beam) {
  std::ofstream file("./outputs/beam_" + std::to_string(beam.ID) + ".csv");
  file << std::fixed << std::setprecision(std::numeric_limits<FPTYPE>::max_digits10);
  for (uint32_t r = 0; r < beam.params.nrays; ++r) {
    auto ray = beam.rays[r];
    file << ray.controls[0] << ", " << ray.controls[1] << ", " << ray.controls[2] << ", " << ray.controls[3] << '\n';
  }
}

int main() {
  uint32_t nx = 201;
  uint32_t ny = 201;

  uint32_t num_beams = 2;

  vec2<FPTYPE> xy_min{-5.0E-4, -5.0E-4}; // xmin and ymin in cm
  vec2<FPTYPE> xy_max{5.0E-4, 5.0E-4}; // xmax and ymax in cm

  auto dx = (xy_max[0] - xy_min[0]) / static_cast<FPTYPE>(nx - 1);
  auto dy = (xy_max[1] - xy_min[0]) / static_cast<FPTYPE>(ny - 1);

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
  auto beam_nrays = static_cast<uint32_t>((2.0 * beam_radius) / dx);

  Parameters params{xy_min, xy_max, cfl, dx, dy, dt, ncrit, nx, ny, nt};
  BeamParams bp0{beam0_norm, beam_radius, beam_sigma, beam_intensity, beam_omega, beam_nrays};
  BeamParams bp1{beam1_norm, beam_radius, beam_sigma, beam_intensity, beam_omega, beam_nrays};

  auto beam0 = new devBeam(bp0, 0);
  auto beam1 = new devBeam(bp1, 1);

  auto ne = new devMatrix<2>(nx, ny);
  linear_electron_density_x<<<1, 1>>>(*ne, ncrit, 0.3, 0.1);
  cudaChk(cudaDeviceSynchronize())

  auto ne_grad = new devVector<2>(nx, ny);
  gradient2D<<<nx, ny>>>(*ne_grad, *ne, dx, dy);
  cudaChk(cudaDeviceSynchronize())

  launch_rays<<<1, beam_nrays>>>(params, *beam0, *ne_grad);
  launch_rays<<<1, beam_nrays>>>(params, *beam1, *ne_grad);
  cudaChk(cudaDeviceSynchronize())

  output_beam(params, *beam0);
  output_beam(params, *beam1);

  delete ne;
  delete ne_grad;
  delete beam0;
  delete beam1;

  return 0;
}
