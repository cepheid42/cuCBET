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

// Usings
using std::pow;
using std::sqrt;

// Constant Values
const float pi = 3.14159265;

int nx = 201;
double xmin = -5.0e-4;
double xmax = 5.0e-4;
double dx = (xmax - xmin) / double(nx - 1);

int nz = 201;
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

//	int rays_per_zone = 5;
auto nrays = 20; //static_cast<int>(double(rays_per_zone) * (beam_max_z - beam_min_z) / dz);

#endif //CUCBET_CONSTANTS_CUH
