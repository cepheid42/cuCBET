#ifndef CUCBET_RAY_CUH
#define CUCBET_RAY_CUH

#include "vec2.cuh"

class Ray: public Managed {
public:
	Ray() : Ray(Point(0.0f, 0.0f), Vec(0.0f, 0.0f), 0.0f) {}

	Ray(const Point& orig, const Vec& dir, float power): orig(orig), dir(dir), power(power) {
		checkErr(cudaMallocManaged(&path,             nt * sizeof(Point)));
		checkErr(cudaMallocManaged(&group_v,          nt * sizeof(Vec)));
		checkErr(cudaMallocManaged(&intersections, nrays * sizeof(int)));
		checkErr(cudaDeviceSynchronize());
	}

	~Ray() {
		checkErr(cudaDeviceSynchronize());
		checkErr(cudaFree(path))
		checkErr(cudaFree(group_v))
		checkErr(cudaFree(intersections));
	}

	Ray& operator=(const Ray* r) {
		if (this != r) {
			orig = r->orig;
			dir = r->dir;
			power = r->power;
			endex = r->endex;

			path = r->path;
			group_v = r->group_v;
			intersections = r->intersections;
		}
		return *this;
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
	int endex = 0;

	Point* path;
	Vec* group_v;
	int* intersections;
};

// Utility Functions
inline bool ray_out_of_range(Point& p) {
	return (p.x() < xmin || p.x() > xmax || p.y() < ymin || p.y() > ymax);
}

#endif //CUCBET_RAY_CUH