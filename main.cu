#include "vec2.cuh"
#include "Beam.cuh"
#include "Egrid.cuh"

#include <chrono>
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;

using std::pow;
using std::array;

void count_intersections(Egrid& eg, Beam& b1) {
	for (auto& r1: b1.rays) {
		for (auto intersection: r1.intersections) {
			int ix = get_index(intersection->x(), xmax, xmin, nx);
			int iz = get_index(intersection->z(), zmax, zmin, nz);

			eg.present[iz][ix] += 1.0;
		}
	}
}

void calculate_gain(Egrid& eg, Beam& b1, Beam& b2, double ncrit) {
	std::cout << "Top of calc_gain." << std::endl;
	for (Ray& ray: b1.rays) {
		std::cout << "looping over rays" << std::endl;
		for (Point* cross_point : ray.intersections) {
			int ix = get_index(cross_point->x(), xmax, xmin, nx);
			int iz = get_index(cross_point->z(), zmax, zmin, nz);

			auto ne = eg.eden[iz][ix];
			auto epsilon = sqrt(1.0 - ne / ncrit);
			auto kmag = (omega / c) * epsilon;

			Point* prev_point1 = get_previous_point(ray, *cross_point);
			Point dk1 = *cross_point - *prev_point1;

			Point* prev_point2;
			for (auto r2: b2.rays) {
				prev_point2 = get_previous_point(r2, *cross_point);
				if (prev_point2){
					break;
				}
			}
			Point dk2 = *cross_point - *prev_point2;

			auto dkmag = dk1.length();

			auto k1 = kmag * (dk1 / (dkmag + 1.0e-10));
			auto k2 = kmag * (dk2 / (dkmag + 1.0e-10));

			auto kiaw = sqrt(pow(k2[0] - k1[0], 2.0) + pow(k2[1] - k1[1], 2.0));
			auto ws = kiaw * cs;

			auto uflow = cs * (2.0 / (xmax - xmin)) * (get_grid_val(ix, xmax, xmin, nx) - xmin) - 2.4;

			double eta = ((omega - omega) - (k2.x() - k1.x()) * uflow) / (ws + 1.0e-10);

			double efield1 = sqrt(8.0 * pi * 1.0e7 * b1.edep[iz][ix] / c);

			double P = (iaw*iaw * eta) / ((eta*eta - 1.0)*(eta*eta - 1.0) + iaw*iaw * eta*eta);
			double gain2 = constant1 * efield1*efield1 * (ne / ncrit) * (1 / iaw) * P;

			// eg.W[0] -> W1
			// eg.W[1] -> W2
			std::cout << "Updating W_new (gain)" << std::endl;
			eg.W_new[iz][ix][0] = eg.W[iz][ix][0] * exp(1 * eg.W[iz][ix][1] * dkmag * gain2 / epsilon);
			eg.W_new[iz][ix][1] = eg.W[iz][ix][1] * exp(-1 * eg.W[iz][ix][0] * dk2.length() * gain2 / epsilon);
		}
	}
}

void calculate_intensity(Egrid& eg, Beam& b1, Beam& b2) {
	std::cout << "Top of calc_intensity" << std::endl;
	b1.edep_new = b1.edep;
	b2.edep_new = b2.edep;

	for (Ray& ray: b1.rays) {
		std::cout << "looping over rays" << std::endl;
		for (int p = 0; p < ray.intersections.size(); p++) {
			Point* cross_point = ray.intersections[p];

			int ix = get_index(cross_point->x(), xmax, xmin, nx);
			int iz = get_index(cross_point->z(), zmax, zmin, nz);

			double frac_change_1 = -1.0 * (1.0 - (eg.W_new[iz][ix][0] / eg.W[iz][ix][0])) * b1.edep[iz][ix];
			double frac_change_2 = -1.0 * (1.0 - (eg.W_new[iz][ix][1] / eg.W[iz][ix][1])) * b2.edep[iz][ix];

			b1.edep_new[iz][ix] += frac_change_1;
			b2.edep_new[iz][ix] += frac_change_2;

			// Increment frac_change for rest of ray
			int ray_ind = get_point_path_index(ray, *cross_point);
			for (int q = ray_ind + 1; q < ray.path.size(); q++) {
				auto next_point = ray.path[q];

				double x_next = next_point.x();
				double z_next = next_point.z();

				int ix_next = get_index(x_next, xmax, xmin, nx);
				int iz_next = get_index(z_next, zmax, zmin, nz);
				std::cout << "Updating W_new (intensity 1)" << std::endl;
				b1.edep_new[iz_next][ix_next] += frac_change_1 * (eg.present[iz][ix] / eg.present[iz_next][ix_next]) ;
			}

			auto b2_point = b2.rays[p];
			int cross_ray_ind = get_point_path_index(b2_point, *cross_point);
			for (int s = cross_ray_ind + 1; s < b2_point.path.size(); s++) {
				auto b2_next_point = b2_point.path[s];

				double x_next = b2_next_point.x();
				double z_next = b2_next_point.z();

				int ix_next = get_index(x_next, xmax, xmin, nx);
				int iz_next = get_index(z_next, zmax, zmin, nz);
				std::cout << "Updating W_new (intensity 2)" << std::endl;
				b2.edep_new[iz_next][ix_next] += frac_change_2 * (eg.present[iz][ix] / eg.present[iz_next][ix_next]);
			}
		}
	}
}

int main() {
	auto start = steady_clock::now();

	double ncrit = 1e-6 * (pow(omega, 2.0) * m_e * e_0 / pow(e_c, 2.0));
	double dt = courant_mult * min(dx, dz) / c;
	auto nt = static_cast<int>(2 * max(nx, nz) / courant_mult);

	/* Todo:
	 * [ ] Figure out gain calculation problems
	 *      [ ] Not reaching print statements
	 *      [ ] Not updating values correctly
	 * [x] Replace all 2d array pointers with array of arrays: double** a -> array<array<double, nx>, nz> a
	 * [x] Figure out electron grid stuff
	 *      [x] Each beam needs its own EDEP for intensity tracking
	 *      [x] Redo file output
	 *
	 * [ ] Add power weighting to rays (intensities and what not)
	 *      [ ] Check electron deposition (make sure it's depositing in correct bins)
	 *
	 * [ ] Determine which code can be parallelized
	 * [ ] Put everything in unified memory for CUDA
	 * [ ] Make parameter struct or something
	 * [ ] Clean up dependencies
	 * [ ] Check for memory leaks
	 * [ ] Optimize
	 *      [ ] use vector Reserve method eg. vec.reserve(100)
	 *      [ ] verify row major operations (been playing fast and loose here)
	*/

	/* Todo: Questions
	 * [ ] Group velocity is updated using d_eden[this_z0, this_x0] for every time step
	 *     should this be updated to be the d_eden[cur_z, cur_x]?
	*/

	Egrid egrid;
	init_egrid(egrid, ncrit);

	Vec beam1_dir(1.0, -0.1);
	Beam b1(0, nrays, beam1_dir);
	double b1_x_start = xmin - (dt / courant_mult * c * 0.5);
	double b1_z_start = beam_min_z + offset - (dz / 2) - (dt / courant_mult * c * 0.5);
	double b1_z_step = (beam_max_z - beam_min_z) / (nrays - 1);
	init_beam(b1, egrid, b1_x_start, b1_z_start, b1_z_step, dt, nt, ncrit);

	Vec beam2_dir(0.0, 1.0);
	Beam b2(1, nrays, beam2_dir);
	double b2_x_start = beam_min_z - (dx / 2) - (dt / courant_mult * c * 0.5);
	double b2_x_step = (beam_max_z - beam_min_z) / (nrays - 1);
	double b2_z_start = zmin - (dt / courant_mult * c * 0.5);
	init_beam(b2, egrid, b2_x_start, b2_z_start, b2_x_step, dt, nt, ncrit);

	count_intersections(egrid, b1);

	calculate_gain(egrid, b1, b2, ncrit);
	calculate_intensity(egrid, b1, b2);

	save_beam_to_file(b1, "beam1");
	save_beam_to_file(b2, "beam2");
	save_egrid_to_files(egrid);

	duration<double> t1 = duration_cast<duration<double>>(steady_clock::now() - start);
	std::cout << "Time: " << t1.count() << " s" << std::endl;
	return 0;
}
