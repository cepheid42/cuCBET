#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"

class Ray {
public:
	~Ray() {
		checkErr(cudaDeviceSynchronize())
		checkErr(cudaFree(path))
		checkErr(cudaFree(group_v))
		checkErr(cudaFree(intersections))
	}

	void allocate(const Point& origin, const Vec& ndir, float uray0) {
		orig = origin;
		dir = ndir;
		power = uray0;
		endex = 0;

		checkErr(cudaMallocManaged(&path,          nt    * sizeof(Point)))
		checkErr(cudaMallocManaged(&group_v,       nt    * sizeof(Vec)))
		checkErr(cudaMallocManaged(&intersections, nrays * sizeof(Point)))
		checkErr(cudaDeviceSynchronize());
	}

	void append_path(Point& p, Vec& v) {
		path[endex] = p;
		group_v[endex] = v;
		endex++;
	}

public:
	Point orig;
	Vec dir;
	float power;
	int endex;

	Point* path{};
	Vec* group_v{};
	Point* intersections{};
};

// Utility Functions
inline bool ray_out_of_range(Point& p) {
	return (p.x < xmin || p.x > xmax || p.y < ymin || p.y > ymax);
}

#endif //CUCBET_RAY_CUH