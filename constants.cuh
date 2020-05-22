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

// Usings
using std::max;
using std::min;
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
double intensity = 2.0e15;
double sigma = 1.7e-4;

int rays_per_zone = 5;
auto nrays = 20; //static_cast<int>(double(rays_per_zone) * (beam_max_z - beam_min_z) / dz);

double uray_mult = intensity * courant_mult / rays_per_zone;


// Utility Functions
inline double get_grid_val(int i, double max, double min, int len) {
	return min + (i * (max - min) / (len - 1));
}

inline int get_index(double v, double max, double min, int len) {
	return static_cast<int>((len - 1) * (v - min) / (max - min));
}

inline double interp(double t, double v0, double v1) {
	return (1 - t) * v0 + t * v1;
}

void save_2d_grid(std::ofstream& file, double** grid, int n_x, int n_z){
	file << std::setprecision(std::numeric_limits<double>::max_digits10);

	for (int i = 0; i < n_z; ++i) {
		for (int j = 0; j < n_x; ++j) {
			file << grid[i][j];
			if (j != n_x - 1) {
				file << ", ";
			}
		}
		file << "\n";
	}
}
#endif //CUCBET_CONSTANTS_CUH
