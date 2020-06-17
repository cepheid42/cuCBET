#ifndef CUCBET_CONSTANTS_CUH
#define CUCBET_CONSTANTS_CUH

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <limits>
#include "Gpu_utils.cuh"
#include "Interpolator.cuh"

// Usings

// Constant Values
const float pi = 3.14159265f;

const int nx = 201;
const float xmin = -5.0e-4f;
const float xmax = 5.0e-4f;
const float dx = (xmax - xmin) / float(nx - 1);

const int ny = 201;
const float ymin = -5.0e-4f;
const float ymax = 5.0e-4f;
const float dy = (ymax - ymin) / float(ny - 1);

const float c = 29979245800.0f;
const float e_0 = 8.85418782e-12f;
const float m_e = 9.10938356e-31f;
const float e_c = 1.60217662e-19f;

const float beam_max = 3.0e-4;
const float beam_min = -3.0e-4;

const float lambda = 1.053e-4f / 3.0f;
const float freq = c / lambda;
const float omega = 2.0f * pi * freq;

const float courant_mult = 0.2f;
const float intensity = 2.0e15f;

const int nrays = int(5 * (beam_max - beam_min) / dy);

const float uray_mult = intensity * courant_mult;

const float estat = 4.80320427e-10f;        // electron charge in statC
const float Z = 3.1f;                       // ionization state
const float mi_kg = 10230.0f * m_e;         // Mass of ion in kg
const float Te = 2.0e3f * 11604.5052f;      // Temperature of electron in K
const float Te_eV = 2.0e3f;
const float Ti = 1.0e3f * 11604.5052f;      // Temperature of ion in K
const float Ti_eV = 1.0e3f;
const float iaw = 0.2f;                     // ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
const float kb = 1.3806485279e-16f;         // Boltzmann constant in erg/K


const float ncrit = 1e-6f * (std::pow(omega, 2.0f) * m_e * e_0 / std::pow(e_c, 2.0f));
const float dt = courant_mult * std::min(dx, dy) / c;
const int nt = int(2.0f * std::max(nx, ny) / courant_mult);

// Utility Functions
__host__ __device__ inline float get_x_val(int i, float max, float min, int N) {
	return min + (float(i) * (max - min) / float(N - 1));
}

__host__ __device__ inline float get_y_val(int i, float max, float min, int N) {
	return max - (float(i) * (max - min) / float(N - 1));
}

__host__ __device__ inline int get_x_index(float x, float max, float min, int N) {
	return int(ceilf(float(N - 1) * (x - min) / (max - min)));
}

__host__ __device__ inline int get_y_index(float y, float max, float min, int N) {
	return int(floorf(float(N - 1) * (y - max) / (min - max)));
}

#endif //CUCBET_CONSTANTS_CUH