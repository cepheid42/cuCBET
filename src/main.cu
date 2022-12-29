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
#include <iomanip>

//#include "Parameters/Parameters.cuh"
#include "./Utilities/Utilities.cuh"
#include "./Beams/Beams.cuh"

template<typename T = float>
std::vector<Vector3<T>> load_beam_normals(const std::string& filename) {
  std::vector<Vector3<T>> beam_normals;

  std::ifstream file(filename);
  std::string line;

  while(std::getline(file, line)) {
    Vector3<T> bnorm;
    std::stringstream lineStream(line);

    for (auto i = 0; i < 3; i++) {
      lineStream >> bnorm[i];
    }
    beam_normals.push_back(bnorm);
  }
  
  return beam_normals;
}

int main() {
  // auto beam_normals = load_beam_normals<float>("./beamnorms.csv");
  // std::vector<Beam*> beams;

  // for (auto i = 0; i < 60; i++) {
  //   beams.push_back(new Beam(i, beam_normals[i], 1.0, 0.1, 10.0, 1.0));
  // }

  auto beam = new Beam(0, Vector3<float>(1.0, 0.0, 0.0), 1.0, 0.1, 10.0, 1.0);

  beam_to_csv(*beam, "./outputs/beam1.csv");

  return 0;
}
