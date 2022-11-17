#ifndef CBET_INTERPOLATOR_CUH
#define CBET_INTERPOLATOR_CUH

#include <cstring>
#include "../Utilities/Utilities.cuh"

/***********************************/
/****** Clamp/Clip functions *******/
template<typename T>
_hd inline constexpr T clip(const T& n, const T& lower, const T& upper) {
  return max(lower, min(n, upper));
}

template<InterpType I, class Manager>
class Interpolator;

/**********************************/
/****** Linear Interpolator *******/
template<class Manager>
class Interpolator<LINEAR, Manager> : public Manager {
public :
  Interpolator(float* xs, float* ys, uint32_t size)
  : size{size}
  {
//    if constexpr(std::is_same<Manager, cpu_managed>::value) {
      xp = new float[size];
      yp = new float[size];
      std::memcpy(xp, xs, size * sizeof(float));
      std::memcpy(yp, ys, size * sizeof(float));
//    } else {
//      cudaChk(cudaMallocManaged(&xp, size * sizeof(float)))
//      cudaChk(cudaDeviceSynchronize())
//      cudaChk(cudaMallocManaged(&yp, size * sizeof(float)))
//      cudaChk(cudaDeviceSynchronize())
//
//      cudaChk(cudaMemcpy(&xp, &xs, size * sizeof(float), cudaMemcpyDefault))
//      cudaChk(cudaMemcpy(&yp, &ys, size * sizeof(float), cudaMemcpyDefault))
//      cudaChk(cudaDeviceSynchronize())
//    }
  }

  float operator()(float x) const {
    // If x is outside range of data,
    // return endpoints, no interpolation possible
    if (x >= xp[size - 1]) {
      std::cout << "too big!" << std::endl;
      return yp[size - 1];
    }
    if (x <= xp[0]) {
      std::cout << "too small!" << std::endl;
      return yp[0];
    }

    // Otherwise find first index with x-value
    // greater than given value
    uint32_t xi;
    for (uint32_t i = 1; i < size - 1; i++) {
      if (x <= xp[i]) {
        std::cout << "Found next biggest at " << i << "(" << x << " < " << xp[i] << ")" << std::endl;
        xi = i;
        break;
      }
    }

    // Get data and calculate deltas
    float x1 = xp[xi];
    float x0 = xp[xi - 1];

    float y1 = yp[xi];
    float y0 = yp[xi - 1];

    // Avoid divide by zero
    float dy = y1 - y0;
    float dx = x1 - x0;
    std::cout << "computing deltas! " << dx << " " << dy << std::endl;
    assert(dx != 0.0f);

    // Compute and return interpolated value
    return y0 + dy * ((x - x0) / dx);
  }

private:
  float* xp;
  float* yp;
  uint32_t size;
};

/************************************/
/****** BiLinear Interpolator *******/
//class BiLinearInterp {
//public:
//  BiLinearInterp(float* xp, float* yp, float* zp, uint32_t xsize, uint32_t ysize)
//  : xp{xp}, yp{yp}, zp{zp},
//    xsize{xsize}, ysize{ysize}
//  {}
//
//  float operator()(float x, float y) const {
//
//  }
//
//private:
//  float* xp;
//  float* yp;
//  float* zp;
//  uint32_t xsize, ysize;
//};

/*************************************/
/****** TriLinear Interpolator *******/
//class TriLinearInterp {
//public:
//  TriLinearInterp(dim3 bounds, float3 mins, float3 deltas, float const* F)
//  : bounds(bounds),
//    mins(mins),
//    deltas(deltas),
//    size(bounds.x * bounds.y * bounds.z),
//    F(F)
//  {
//    // Need at least 2x2x2 points to work
//    logAssert((bounds.x >= 2) && (bounds.y >= 2) && (bounds.z >= 2), "Invalid input bounds.");
//    logAssert((deltas.x > 0.0) && (deltas.y > 0.0) && (deltas.z > 0), "Invalid input delta.");
//    logAssert((F != nullptr), "Invalid function pointer.");
//
//    maxs.x = mins.x + deltas.x * static_cast<float>(bounds.x) - static_cast<float>(1);
//    maxs.y = mins.y + deltas.y * static_cast<float>(bounds.y) - static_cast<float>(1);
//    maxs.z = mins.z + deltas.z * static_cast<float>(bounds.z) - static_cast<float>(1);
//
//    invDeltas = make_float3(1.0 / deltas.x, 1.0 / deltas.y, 1.0 / deltas.z);
//  }
//
//  __host__ float operator() (float x, float y, float z) const {
//    // Compute x-index and clamp to image?
//    auto xInd = (x - mins.x) * invDeltas.x;
//    auto yInd = (y - mins.y) * invDeltas.y;
//    auto zInd = (z - mins.z) * invDeltas.z;
//
//    auto xi = clip<uint32_t>(static_cast<uint32_t>(xInd), 0, bounds.x);
//    auto yi = clip<uint32_t>(static_cast<uint32_t>(yInd), 0, bounds.y);
//    auto zi = clip<uint32_t>(static_cast<uint32_t>(zInd), 0, bounds.z);
//
//    float U[2] = {1.0, xInd - static_cast<float>(xi)};
//    float V[2] = {1.0, yInd - static_cast<float>(yi)};
//    float W[2] = {1.0, zInd - static_cast<float>(zi)};
//    float P[2], Q[2], R[2];
//
//    // Compute P = MU, Q = MV, R = MW
//    for(auto row = 0; row < 2; row++) {
//      P[row] = 0;
//      Q[row] = 0;
//      R[row] = 0;
//
//      for(auto col = 0; col < 2; col++) {
//        P[row] += blend[row][col] * U[col];
//        Q[row] += blend[row][col] * V[col];
//        R[row] += blend[row][col] * W[col];
//      }
//    }
//
//    // Compute tensor product (MU)(MV)(MW)D, where D is 2x2x2 subimage containing (x, y, z)
//    float result = 0.0;
//
//    for(uint32_t slice = 0; slice < 2; slice++) {
//      auto zclamp = clip<uint32_t>(zi + slice, 0, bounds.z)
//    }
//  }
//
//private:
//  dim3 bounds;
//  float3 mins;
//  float3 maxs;
//  float3 deltas;
//  float3 invDeltas;
//  uint32_t size;
//  float const* F;
//  float blend[2][2];
//};

#endif //CBET_INTERPOLATOR_CUH
