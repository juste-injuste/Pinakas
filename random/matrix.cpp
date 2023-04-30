#include "matrix.hpp"
#include <float.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
using std::cout;
using std::endl;


matrix matrix::operator | (matrix b) {		// solve linear system using QR decomposition and back-substitution
	unsigned int   N = size.x, M = size.y;
	double V[N][M];
	double Q[N][M];
	double R[N][N];
	double Qtb[N] = {0};
	matrix x(1, N);
	unsigned int i, j, k;
	double proj;
	double norm, inorm;

	for (j=0; j<N; j++) {		// QR decomposition using modified Gram-Schmidt process
		for (i=0; i<M; i++) {
			V[j][i] = data[j + i*N];
		}
	}
	for (j=0; j<N; j++) {
		proj = 0;
		for (i=0; i<M; i++) {
			proj += V[j][i] * V[j][i];
		}
		norm = sqrt(proj);
		inorm = 1/norm;
		for (i=0; i<M; i++) {
			Q[j][i] = V[j][i] * inorm;
		}
		R[j][j] = norm;
		for (k=j+1; k<N; k++) {
			proj = 0;
			for (i=0; i<M; i++) {
				proj += Q[j][i] * V[k][i];
			}
			for (i=0; i<M; i++) {
				V[k][i] -= Q[j][i] * proj;
			}
			R[k][j] = proj;
		}
	}

	for (i=0; i<N; i++) {		// compute Qt * b
		for (j=0; j<M; j++) {
			Qtb[i] += Q[i][j] * b.data[j];
		}
	}
	
	for (i=N-1; i<N; i--) {	// solve Rx = Qtb using back-substitution
		x.data[i] = Qtb[i];
		for (j=N-1; j>i; j--) {
			x.data[i] -= R[j][i] * x.data[j];
		}
		x.data[i] /= R[i][i];
	}

	return x;
}

