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
//const float offset = 0.5e-4f;
const float intensity = 2.0e15f;
const float sigma = 1.7e-4f;

const int rays_per_zone = 5;
const auto nrays = 25; //static_cast<int>(float(rays_per_zone) * (beam_max - beam_min) / dy);

const float uray_mult = intensity * courant_mult / float(rays_per_zone);

const float estat = 4.80320427e-10f;      // electron charge in statC
const float Z = 3.1f;                     // ionization state
//const float mi = 10230.0 * 1.0e3 * m_e;  // Mass of ion in g
const float mi_kg = 10230.0f * m_e;       // Mass of ion in kg
const float Te = 2.0e3f * 11604.5052f;     // Temperature of electron in K
const float Te_eV = 2.0e3f;
const float Ti = 1.0e3f * 11604.5052f;     // Temperature of ion in K
const float Ti_eV = 1.0e3f;
const float iaw = 0.2f;                   // ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
const float kb = 1.3806485279e-16f;       // Boltzmann constant in erg/K
//const float kb2 = 1.3806485279e-23f;      // Boltzmann constant in J/K


float constant1 = std::pow(estat, 2.0f) / (4.0f * (1.0e3f * m_e) * c * omega * kb * Te * (1.0f + 3.0f * Ti / (Z * Te)));
float cs = 100.0f * std::sqrt(e_c * (Z * Te_eV + 3.0f * Ti_eV) / mi_kg);


// Utility Functions
/*
 * x is negative->positive, left->right, y is negative->positive, bottom->top
 * in order to map coordinates, the y functions are reversed from the x functions.
 *
 * Use the X functions for normal mapping.
 */
inline float get_x_val(int i, float max, float min, int N) {
	return min + (float(i) * (max - min) / float(N - 1));
}

inline float get_y_val(int i, float max, float min, int N) {
	return max - (float(i) * (max - min) / float(N - 1));
}

inline int get_x_index(float x, float max, float min, int N) {
	return int(std::ceil(float(N - 1) * (x - min) / (max - min)));
}

inline int get_y_index(float y, float max, float min, int N) {
	return int(std::floor(float(N - 1) * (y - max) / (min - max)));
}

#endif //CUCBET_CONSTANTS_CUH