#include "matrix.hpp"
#include <float.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
using std::cout;
using std::endl;

matrix matrix::operator , (matrix arr) {
	unsigned int new_x = size.x+arr.size.x;
	unsigned int i, j;
	matrix res(new_x, size.y);

	for (i=0; i<new_x; i++)
		for (j=0; j<size.y; j++)
			res(i, j) = i<size.x ? data[i + j*size.x] : arr(i-size.x, j);
	
	return res;
}

matrix matrix::operator , (double val) {
	matrix res = *this;
	res.resize(size.numel+1);
	res[size.numel] = val;	
	return res;
}

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

matrix matrix::operator ^ (double val) {
	matrix res(size);
	unsigned int i;
	for (i=0; i<size.numel; i++)
		res[i] = pow(data[i], val);
	return res;
}


matrix::matrix(void) {
	size 	= {0, 0, 0};
	data   	= (double*) calloc(0, sizeof(double));
};

matrix::matrix(unsigned int xsz, unsigned int ysz) {
	size 	= {xsz, ysz, xsz * ysz};
	data   	= (double*) calloc(size.numel, sizeof(double));
};

matrix::matrix(_size sz) {
	size 	= sz;
	data   	= (double*) calloc(size.numel, sizeof(double));
};

matrix::matrix(unsigned int sz) {
	size 	= {1, sz, sz};
	data   	= (double*) calloc(size.numel, sizeof(double));
};


matrix matrix::operator & (matrix B) {
	matrix res(B.size.x, size.y);
	long double sum;
	unsigned int i, j, k;
	for (i=0; i<B.size.x; i++) {
		for (j=0; j<size.y; j++) {
			sum = 0;
			for (k=0; k<size.x; k++) {
				sum += data[k + j*size.x] * B[i + k*B.size.x];
			}
			res(i, j) = sum;
		}
	}
	return res;
}

matrix matrix::operator ! (void) {	// transpose matrix
	matrix res(size.y, size.x);
	unsigned i, j;
	for (i=0; i<size.x; i++) {
		for (j=0; j<size.y; j++) {
			res(j, i) = data[i + j*size.x];
		}
	}
	return res;
}



inline double& matrix::operator () (unsigned int x_idx, unsigned int y_idx) {
	if (x_idx + 1 > size.x) resize(x_idx+1, size.y);
	if (y_idx + 1 > size.y) resize(size.x, y_idx+1);
	return data[x_idx + y_idx*size.x];
}

inline double& matrix::operator () (unsigned int idx) {
	if (idx + 1 > size.numel) resize(idx+1);
	return data[idx];
}

inline double& matrix::operator [] (unsigned int idx) {
	return data[idx];
}

matrix matrix::operator = (matrix arr) {
	size 	= arr.size;
	double* temp = (double*) calloc(size.numel, sizeof(double));
	unsigned int i;
	free(data);
	for (i=0;i<size.numel;i++) {
		temp[i] = arr.data[i];
	}
	data 	= temp;
	return *this;
}

matrix matrix::operator + (matrix arr) {
	matrix res(size);
	unsigned int i;
	for (i=0;i<size.numel;i++) {
		res.data[i] = data[i] + arr.data[i];
	}
	return res;
}

matrix matrix::operator + (double val) {
	matrix res(size);
	unsigned int i;
	for (i=0;i<size.numel;i++)
		res.data[i] = data[i] + val;
	return res;
}

matrix matrix::operator += (matrix arr) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] += arr.data[i];
	return *this;
}

matrix matrix::operator += (double val) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] += val;
	return *this;
}

matrix matrix::operator - (void) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] = -data[i];
	return *this;
}

matrix matrix::operator - (matrix arr) {
	matrix res(size);
	unsigned int i;
	for (i=0;i<size.numel;i++)
		res.data[i] = data[i] - arr.data[i];
	return res;
}

matrix matrix::operator - (double val) {
	matrix res(size);
	unsigned int i;
	for (i=0;i<size.numel;i++)
		res.data[i] = data[i] - val;
	return res;
}

matrix matrix::operator -= (matrix arr) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] -= arr.data[i];
	return *this;
}

matrix matrix::operator -= (double val) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] -= val;
	return *this;
}
matrix matrix::operator * (matrix arr) {
	matrix res(size);
	unsigned int i;
	for (i=0;i<size.numel;i++)
		res.data[i] = data[i] * arr.data[i];
	return res;
}

matrix matrix::operator * (double val) {
	matrix res(size);
	unsigned int i;
	for (i=0;i<size.numel;i++)
		res.data[i] = data[i] * val;
	return res;
}


matrix matrix::operator *= (matrix arr) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] *= arr.data[i];
	return *this;
}

matrix matrix::operator *= (double val) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] *= val;
	return *this;
}
matrix matrix::operator / (matrix arr) {
	matrix res(size);
	unsigned int i;
	for (i=0;i<size.numel;i++)
		res.data[i] = data[i] / arr.data[i];
	return res;
}

matrix matrix::operator / (double val) {
	matrix res(size);
	unsigned int i;
	for (i=0;i<size.numel;i++)
		res.data[i] = data[i] / val;
	return res;
}

matrix matrix::operator /= (matrix arr) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] /= arr.data[i];
	return *this;
}

matrix matrix::operator /= (double val) {
	unsigned int i;
	for (i=0;i<size.numel;i++)
		data[i] /= val;
	return *this;
}

matrix matrix::show(void) {
	unsigned int i, j;
	cout << "array = ";
	for (j=0; j<size.y ; j++) {
		cout << endl;
		for (i=0; i<size.x; i++) {
			cout << std::setw(13) << std::setprecision(6) << data[i+j*size.x];
		}
	}
	cout << endl;
	return *this;
}

int matrix::resize(unsigned int xsz, unsigned int ysz) {
	double* temp = (double*) calloc(xsz*ysz, sizeof(double));
	unsigned int dif = xsz - size.x;
	unsigned int i, j;
	for (i=0, j=0; i<size.numel; i++, j++) {
		temp[j] = data[i];
		j += dif*!((i+1)%size.x);
	}

	free(data);
	data 	= temp;
	size  	= {xsz, ysz, xsz * ysz};
	return  0;
}

int matrix::resize(_size sz) {
	double* temp = (double*) calloc(sz.numel, sizeof(double));
	unsigned int dif = sz.x - size.x;
	unsigned int i, j;
	for (i=0, j=0; i<size.numel; i++, j++) {
		temp[j] = data[i];
		j += dif*!((i+1)%size.x);
	}

	free(data);
	data 	= temp;
	size  	= sz;
	return  0;
}


