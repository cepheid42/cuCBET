//uint32_t* marked  = 200 * nx * ny * nz * sizeof(uint32_t);
//uint32_t* counter = nx * ny * nz * sizeof(uint32_t);
//float* eden       = nx * ny * nz * sizeof(float);
//float* etemp      = nx * ny * nz * sizeof(float);
//float* machnum    = 3 * nx * ny * nz * sizeof(float);
//
//float* intensity_init = nbeams * nrays * sizeof(float);
//float* kvec           = nbeams * nrays * ncrossings * sizeof(float);
//float* i_b            = nbeams * nrays * ncrossings * sizeof(float);
//float* ray_areas      = nbeams * nrays * ncrossings * sizeof(float);
//float* polar_angle    = nbeams * nrays * ncrossings * sizeof(float);
//float* wMult          = 2 * nbeams * nrays * ncrossings * sizeof(float);



//#include "Parameters/Parameters.cuh"
#include "./Utilities/Utilities.cuh"
#include "./Beams/Beams.cuh"


int main() {
  cpuTimer timer;
  timer.start();

  auto beam_normals = load_beam_normals<float>("./beamnorms.csv");
  std::vector<Beam*> beams;

  for (auto i = 0; i < 60; i++) {
    beams.push_back(new Beam(i, beam_normals[i], 1.0, 0.1, 10.0, 1.0));
  }

  timer.stop();
  std::cout << "Time: " << timer.elapsed() << std::endl;


  // int nx = 100;
  // int ny = nx;
  // int nz = nx;

  // float xmin = -0.0013; // meters
  // float xmax = 0.0013; // meters

  // float dx = (xmax - xmin) / float(nx - 1);
  // float dy = dx;
  // float dz = dx;

  // float beam_radius = 2.0E-6; // meters

  // float lambda_beam = 3.51E-7 // meters (351 nm)
  // float freq_beam = Constants::C0 / lambda;
  // float omega_beam = 2.0 * Constants::PI * freq_beam;
  // float sigma_beam = 0.0375;
  // float I0_beam = 1.0;

  // float ncrit = 1.0E-6 * SQR(omega_beam) * Constants::Me * Constants::EPS0 / SQR(Constants::qe);

  // float mach_max = 2.4;
  // float mach_min = 0.4;

  // float eDen_max = 0.3;
  // float eDen_min = 0.1;
  // float eDen_step = (eDen_max - eDen_min) / 


  return 0;
}
