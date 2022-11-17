#ifndef CBET_PARAMETERS_CUH
#define CBET_PARAMETERS_CUH

#include "../Utilities/Utilities.cuh"

//#ifdef USE_OMEGA
//auto spacesize = 5E-4;
//auto beam_min_x = -2E-4;
//auto beam_max_x = 2E-4;
//auto nbeams = 3;
//auto beam_stddev = 2;
//auto intensity = 1E17;
//#else
//auto spacesize = 0.13;
//auto beam_min_x = -450E-4;
//auto beam_max_x = 450E-4;
//auto nbeams = 2;
//auto beam_stddev = 1;
//auto intensity = 1E14;
//#endif

class Parameters {
public:
  Parameters(dim3 Ldims, float3 Pmax, float3 Pmin, float CFL_r, uint32_t nBeams)
  : Ldims(Ldims),
    Pmax(Pmax),
    Pmin(Pmin),
    cfl_r(CFL_r),
    nBeams(nBeams)
  {

    auto dx = (Pmax.x - Pmin.x) / static_cast<float>(Ldims.x - 1);
    auto dy = (Pmax.y - Pmin.y) / static_cast<float>(Ldims.y - 1);
    auto dz = (Pmax.z - Pmin.z) / static_cast<float>(Ldims.z - 1);
    deltas = make_float3(dx, dy, dz);

    const auto delta = std::sqrt(SQR(dx) + SQR(dy) + SQR(dz));
    dt = (CFL_r * delta) / (3.0f * Constants::C0);
  }

  float dx() const { return deltas.x; }
  float dy() const { return deltas.y; }
  float dz() const { return deltas.z; }

  uint32_t nx() const { return Ldims.x; }
  uint32_t ny() const { return Ldims.y; }
  uint32_t nz() const { return Ldims.z; }

  float minX() const  { return Pmin.x; }
  float minY() const  { return Pmin.y; }
  float minZ() const  { return Pmin.z; }

  float maxX() const  { return Pmax.x; }
  float maxY() const  { return Pmax.y; }
  float maxZ() const  { return Pmax.z; }

protected:
  dim3 Ldims;
  float3 deltas;
  float3 Pmax;
  float3 Pmin;
  float dt;
  float cfl_r;
  uint32_t nBeams;
};

#endif //CBET_PARAMETERS_CUH
