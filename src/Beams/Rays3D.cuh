#ifndef CBET_RAYS_CUH
#define CBET_RAYS_CUH

#include "../Utilities/Utilities.cuh"

struct Ray {
  float controls[9];
  float intensity;

  Ray() = default;
  
  _hd Ray(const Vector3<float>& A, const Vector3<float>& B, const Vector3<float>& C, float _intensity) 
  : controls{A[0], B[0], C[0], 
             A[1], B[1], C[1], 
             A[2], B[2], C[2]},
    intensity(_intensity)
  {}

  _hd Vector3<float> mins() const {
    auto x = min(controls[0], min(controls[1], controls[2]));
    auto y = min(controls[3], min(controls[4], controls[5]));
    auto z = min(controls[6], min(controls[7], controls[8]));
    return {x, y, z};
  }
  
  _hd Vector3<float> maxs() const {
    auto x = max(controls[0], max(controls[1], controls[2]));
    auto y = max(controls[3], max(controls[4], controls[5]));
    auto z = max(controls[6], max(controls[7], controls[8]));
    return {x, y, z};
  }

  // Update end control point
  _hd void update_C_control(const Vector3<float>& newC) {
    controls[2] = newC[0];
    controls[5] = newC[1];
    controls[8] = newC[2];
  }

  // Evaluate P(t) = (1 - t)^2 * A + 2t(1-t) * B + t^2 * C
  _hd Vector3<float> eval(float t) const {
    float omt = 1.0 - t;
    float px = (omt * omt * controls[0]) + (2.0 * t * omt * controls[1]) + (t * t * controls[2]);
    float py = (omt * omt * controls[3]) + (2.0 * t * omt * controls[4]) + (t * t * controls[5]);
    float pz = (omt * omt * controls[6]) + (2.0 * t * omt * controls[7]) + (t * t * controls[8]);

    return {px, py, pz};
  }
};

enum DIR { x = 0, y = 3, z = 6};

template<DIR dir>
_hd bool threeway_compare(const Ray& u, const Ray& v) {
  auto u0 = min(u.controls[dir], min(u.controls[dir + 1], u.controls[dir + 2]));
  auto u1 = max(u.controls[dir], max(u.controls[dir + 1], u.controls[dir + 2]));

  auto v0 = min(v.controls[dir], min(v.controls[dir + 1], v.controls[dir + 2]));
  auto v1 = max(v.controls[dir], max(v.controls[dir + 1], v.controls[dir + 2]));

  auto lb = max(u0, v0);
  auto ub = min(u1, v1);

  // Does this need to be more robust
  // for comparing floats/doubles?
  // eg. (a - b) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon)
  if (lb > ub) {
    return false;
  }

  return true;
}

_hd bool rays_bb_intersect(const Ray& u, const Ray& v) {
  if (!threeway_compare<DIR::x>(u, v)) {
    return false;
  }

  if (!threeway_compare<DIR::y>(u, v)) {
    return false;
  }

  if (!threeway_compare<DIR::z>(u, v)) {
    return false;
  }

  return true;
}


#endif //CBET_RAYS_CUH
