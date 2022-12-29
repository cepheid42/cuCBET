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

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

//#include "Parameters/Parameters.cuh"
#include "./Utilities/Utilities.cuh"
#include "./Beams/Beams.cuh"

template<typename T = float>
std::vector<std::vector<T>> load_beam_normals(const std::string& filename) {
  std::vector<std::vector<T>> beam_normals;

  std::ifstream file(filename);
  std::string line;
  // Read one line at a time into the variable line:
  while(std::getline(file, line)) {
      std::vector<T> lineData;
      std::stringstream lineStream(line);
      T value;

      while(lineStream >> value) {
          lineData.push_back(value);
      }
      beam_normals.push_back(lineData);
  }
  
  return beam_normals;
}

int main() {
  auto beam_normals = load_beam_normals("./beamnorms.csv");

  for (auto bnorm: beam_normals) {
    std::cout << bnorm << std::endl;
  }

  return 0;
}
