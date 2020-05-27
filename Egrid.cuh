#ifndef CUCBET_EGRID_CUH
#define CUCBET_EGRID_CUH

#include "vec2.cuh"

using std::array;

class Egrid {
public:
	Egrid() = default;

public:
	array<array<float, nx>, ny> eden{};
	array<array<Point, nx>, ny> d_eden;
	array<array<Point, nx>, ny> W;
	array<array<Point, nx>, ny> W_new;
};

void init_eden_derivs(Egrid& eg) {
	// This function may possible be parallelizable...
	for (int y = 0; y < ny - 1; ++y) {
		for (int x = 0; x < nx - 1; ++x) {
			eg.d_eden[y][x][0] = (eg.eden[y][x + 1] - eg.eden[y][x]) / (get_x_val(x + 1, xmax, xmin, nx) - get_x_val(x, xmax, xmin, nx));
			eg.d_eden[y][x][1] = (eg.eden[y + 1][x] - eg.eden[y][x]) / (get_y_val(y + 1, ymax, ymin, ny) - get_y_val(y, ymax, ymin, ny));
		}
	}

	// set last column equal to previous column
	for (int y = 0; y < ny; y++) {
		eg.d_eden[y][nx - 1] = eg.d_eden[y][nx - 2];
	}

	// set last row equal to previous row
	for (int x = 0; x < nx; x++) {
		eg.d_eden[ny - 1][x] = eg.d_eden[ny - 2][x];
	}
}

void init_egrid(Egrid& eg, float ncrit) {
	for (int y = 0; y < ny; ++y) {
		for (int x = 0; x < nx; ++x) {
			float density = ((0.3f * ncrit - 0.1f * ncrit) / (xmax - xmin)) * (get_x_val(x, xmax, xmin, nx) - xmin) + (0.1f * ncrit);
			eg.eden[y][x] = std::max(density, 0.0f);

			float w = std::sqrt(1.0f - eg.eden[y][x] / ncrit) / float(rays_per_zone);
			eg.W[y][x] = Point(w, w);
		}
	}
	eg.W_new = eg.W;
	init_eden_derivs(eg);
}

// Utility Functions
void save_egrid_to_files(Egrid& eg) {
	std::ofstream eden_file("eden.csv");
	eden_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (int i = ny - 1; i >= 0; i--) {
		for (int j = 0; j < nx; j++) {
			eden_file << eg.eden[i][j];
			if (j != nx - 1) {
				eden_file << ", ";
			}
		}
		eden_file << "\n";
	}
	eden_file.close();

	std::ofstream i_b1_file("i_b1.csv");
	std::ofstream i_b2_file("i_b2.csv");
	i_b1_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	i_b2_file << std::setprecision(std::numeric_limits<float>::max_digits10);

	for (int k = ny - 1; k >= 0; k--) {
		for (int l = 0; l < nx; l++) {
			i_b1_file << eg.W[k][l][0] << ", ";
			i_b2_file << eg.W[k][l][1] << ", ";
		}
		i_b1_file << "\n";
		i_b2_file << "\n";
	}
	i_b1_file.close();
	i_b2_file.close();

	std::ofstream i_b1_new_file("i_b1_new.csv");
	std::ofstream i_b2_new_file("i_b2_new.csv");
	i_b1_new_file << std::setprecision(std::numeric_limits<float>::max_digits10);
	i_b2_new_file << std::setprecision(std::numeric_limits<float>::max_digits10);

	for (int q = ny - 1; q >= 0; q--) {
		for (int p = 0; p < nx; p++) {
			i_b1_new_file << eg.W_new[q][p][0] << ", ";
			i_b2_new_file << eg.W_new[q][p][1] << ", ";
		}
		i_b1_new_file << "\n";
		i_b2_new_file << "\n";
	}
	i_b1_new_file.close();
	i_b2_new_file.close();
}

#endif //CUCBET_EGRID_CUH
