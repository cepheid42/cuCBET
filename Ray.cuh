#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"

struct Intersection {
	int r1_num;
	int r1_ind;
	int r2_num;
	int r2_ind;
};

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
		int_index = 0;

		checkErr(cudaMallocManaged(&path,          nt    * sizeof(Point)))
		checkErr(cudaMallocManaged(&group_v,       nt    * sizeof(Vec)))
		checkErr(cudaMallocManaged(&intersections, nrays * sizeof(Intersection)))
		checkErr(cudaDeviceSynchronize());
	}

	void append_path(Point& p, Vec& v) {
		path[endex] = p;
		group_v[endex] = v;
		endex++;
	}

	__device__ void add_intersection(int ray1_num, int ray1_ind, int ray2_num, int ray2_ind) const {
		intersections[ray2_num].r1_num = ray1_num;
		intersections[ray2_num].r1_ind = ray1_ind;
		intersections[ray2_num].r2_num = ray2_num;
		intersections[ray2_num].r2_ind = ray2_ind;
	}

public:
	Point orig;
	Vec dir;
	float power;
	int endex;
	int int_index;

	Point* path{};
	Vec* group_v{};
	Intersection* intersections{};
};

// Utility Functions
inline bool ray_out_of_range(Point& p) {
	return (p.x < xmin || p.x > xmax || p.y < ymin || p.y > ymax);
}

#endif //CUCBET_RAY_CUH