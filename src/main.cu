#include "Utilities/Utilities.cuh"

#include "Beams/Beam2D.cuh"

int main() {
  uint32_t nx = 201;
  uint32_t ny = 201;

  FPTYPE xmax = 5.0E-4; // 5 microns
  FPTYPE ymax = xmax;
  FPTYPE xmin = -xmax;
  FPTYPE ymin = xmin;

  auto dx = (xmax - xmin) / static_cast<FPTYPE>(nx - 1);
  auto dy = (ymin - ymax) / static_cast<FPTYPE>(ny - 1);

  auto cfl = 1.0 / sqrt(2.0);

  auto nt = nx / cfl;

  FPTYPE lambda = 3.5E-5; // cm



  return 0;
}
