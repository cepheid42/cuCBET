#ifndef CUCBET_CONSTANTS_CUH
#define CUCBET_CONSTANTS_CUH

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cassert>
#include <array>

// Usings


// Constant Values
const float pi = 3.14159265f;

const int nx = 201;
float xmin = -5.0e-4f;
float xmax = 5.0e-4f;
float dx = (xmax - xmin) / float(nx - 1);

const int ny = 201;
float ymin = -5.0e-4f;
float ymax = 5.0e-4f;
float dy = (ymax - ymin) / float(ny - 1);

float c = 29979245800.0f;
float e_0 = 8.85418782e-12f;
float m_e = 9.10938356e-31f;
float e_c = 1.60217662e-19f;

float beam_max = 3.0e-4;
float beam_min = -3.0e-4;

float lambda = 1.053e-4f / 3.0f;
float freq = c / lambda;
float omega = 2.0f * pi * freq;

float courant_mult = 0.2f;
float offset = 0.5e-4f;
float intensity = 2.0e15f;
float sigma = 1.7e-4f;

int rays_per_zone = 5;
auto nrays = 25; //static_cast<int>(float(rays_per_zone) * (beam_max_z - beam_min_z) / dz);

float uray_mult = intensity * courant_mult / float(rays_per_zone);

float estat = 4.80320427e-10f;      // electron charge in statC
float Z = 3.1f;                     // ionization state
//float mi = 10230.0 * 1.0e3 * m_e;  // Mass of ion in g
float mi_kg = 10230.0f * m_e;       // Mass of ion in kg
float Te = 2.0e3f * 11604.5052f;     // Temperature of electron in K
float Te_eV = 2.0e3f;
float Ti = 1.0e3f * 11604.5052f;     // Temperature of ion in K
float Ti_eV = 1.0e3f;
float iaw = 0.2f;                   // ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
float kb = 1.3806485279e-16f;       // Boltzmann constant in erg/K
float kb2 = 1.3806485279e-23f;      // Boltzmann constant in J/K
float constant1 = std::pow(estat, 2.0f) / (4.0f * (1.0e3f * m_e) * c * omega * kb * Te * (1.0f + 3.0f * Ti / (Z * Te)));

float cs = 100.0f * std::sqrt(e_c * (Z * Te_eV + 3.0f * Ti_eV) / mi_kg);

		
		
// Utility Functions
inline float get_grid_val(int i, float max, float min, int len) {
	return min + (float(i) * (max - min) / float(len - 1));
}

inline int get_x_index(float x, float max, float min, int N) {
	return int(float(N - 1) * (x - min) / (max - min));
}

inline int get_y_index(float x, float max, float min, int N) {
	return int(float(N - 1) * (x - max) / (min - max));
}

inline float interp(float x, float x0, float y0, float x1, float y1) {
	return (y0 + (x - x0) * (y1 - y0) / (x1 - x0));
}

#endif //CUCBET_CONSTANTS_CUH
