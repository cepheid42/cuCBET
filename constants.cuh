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
using std::max;
using std::min;
using std::pow;
using std::sqrt;

// Constant Values
const float pi = 3.14159265;

const int nx = 201;
double xmin = -5.0e-4;
double xmax = 5.0e-4;
double dx = (xmax - xmin) / double(nx - 1);

const int nz = 201;
double zmin = -5.0e-4;
double zmax = 5.0e-4;
double dz = (zmax - zmin) / double(nz - 1);

double c = 29979245800.0;
double e_0 = 8.85418782e-12;
double m_e = 9.10938356e-31;
double e_c = 1.60217662e-19;

double beam_max_z = 3.0e-4;
double beam_min_z = -3.0e-4;

double lambda = 1.053e-4 / 3.0;
double freq = c / lambda;
double omega = 2 * pi * freq;

double courant_mult = 0.2;
double offset = 0.5e-4;
double intensity = 2.0e15;
double sigma = 1.7e-4;

int rays_per_zone = 5;
auto nrays = static_cast<int>(double(rays_per_zone) * (beam_max_z - beam_min_z) / dz);

double uray_mult = intensity * courant_mult / rays_per_zone;

double estat = 4.80320427e-10;      // electron charge in statC
double Z = 3.1;                     // ionization state
double mi = 10230 * (1.0e3 * m_e);  // Mass of ion in g
double mi_kg = 10230.0 * m_e;       // Mass of ion in kg
double Te = 2.0e3 * 11604.5052;     // Temperature of electron in K
double Te_eV = 2.0e3;
double Ti = 1.0e3 * 11604.5052;     // Temperature of ion in K
double Ti_eV = 1.0e3;
double iaw = 0.2;                   // ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
double kb = 1.3806485279e-16;       // Boltzmann constant in erg/K
double kb2 = 1.3806485279e-23;      // Boltzmann constant in J/K
double constant1 = pow(estat, 2.0) / (4 * (1.0e3 * m_e) * c * omega * kb * Te * (1 + 3 * Ti / (Z * Te)));

double cs = 1e2 * sqrt(e_c * (Z * Te_eV + 3.0 * Ti_eV) / mi_kg);
		
		
// Utility Functions
inline double get_grid_val(int i, double max, double min, int len) {
	return min + (i * (max - min) / (len - 1));
}

inline int get_index(double v, double max, double min, int len) {
	return static_cast<int>((len - 1) * (v - min) / (max - min));
}

inline double interp(double x, double x0, double y0, double x1, double y1) {
	return (y0 * (1 - (x - x0) / (x1 - x0))) + (y1 * ((x - x0) / (x1 - x0)));
}

#endif //CUCBET_CONSTANTS_CUH
