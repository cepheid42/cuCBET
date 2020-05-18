#ifndef CUCBET_EGRID_CUH
#define CUCBET_EGRID_CUH

class Egrid {
public:
	Egrid() = default;
	Egrid(int nx, int nz) : x(nx), z(nz) {
		eden = new float*[x];
		for (int i = 0; i < x; ++i) {
			eden[i] = new float[z];
		}

		edep = new float*[x + 2];
		for (int i = 0; i < x + 2; ++i) {
			edep[i] = new float[z + 2];
		}
	}

public:
	int x;
	int z;
	float **eden;
	float **edep;
};

void init_eden(Egrid& e, float x_range) {
	float ncrit = 0.2;

	for (int i = 0; i < e.x; ++i) {
		for (int j = 0; j < e.z; ++j) {
			auto val = ((0.2 * ncrit) / (x_range)) *
			e.eden[i][j] =
		}
	}
}

#endif //CUCBET_EGRID_CUH
