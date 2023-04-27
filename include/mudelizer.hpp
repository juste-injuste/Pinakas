#ifndef mudelizer_hpp
#define mudelizer_hpp
#include "matrix.hpp"


matrix knee(matrix arr, double n);

matrix_array MGS(matrix A);




_idxval matrix::min(void) {
	double min = DBL_MAX;
	int idx = size.numel;
	for (int i=0; i<size.numel; i++) {
		if (data[i] < min) {
			min = data[i];
			idx = i;
		}
	}
	return  (_idxval) {idx, min};
}



matrix sin(matrix arr) {
	matrix res(arr.sizex(), arr.sizey());
	for (int i=0; i<arr.numel(); i++)
		res[i] = sin(arr[i]);
	return res;
}




matrix knee(matrix arr, double n) {
    matrix res(arr.sizex(), arr.sizey());
    for (int i=0;i<arr.numel();i++)
		res[i] = 1/(arr[i]+n);
    return res;
}

long double sum(matrix arr) {
	long double sum = 0;
	for (int i=0; i<arr.numel(); i++)
		sum += arr[i];
	return sum;
}



double err(matrix g, matrix y) {
	long double sum = 0;
	for (int i=0; i<y.sizex(); i++) {
		for (int j=0; j<y.sizey(); j++) {
			sum += g(i, j)*g(i, j) - 2*g(i, j)*y(i, j)  + y(i, j)*y(i, j);
		}
	}
	return fabs(sum);
}



double newton(double (*f)(double), double seed=1, double tol=1e-6, int max=10) {
	double tol_2 	= tol*0.5;
	double root 	= seed;

	for (int i=0; fabs(f(root)) > tol && i < max; i++) {
		root	-= tol*f(root)/(f(root+tol_2)-f(root-tol_2));
		//cout << i << endl;
	}

	return root;
}




matrix blackman(unsigned int L) {
	matrix window(L);
	for (int n=0; n<L; n++) {
		window[n] = 0.42 - 0.5*cos(2*M_PI*n/(L-1)) + 0.08*cos(4*M_PI*n/(L-1));
	}
	return window;
}

matrix hamming(unsigned int L) {
	matrix window(L);
	for (int n=0; n<L; n++) {
		window[n] = 0.54 - 0.46*cos(2*M_PI*n/(L-1));
	}
	return window;
}

matrix hann(unsigned int L) {
	matrix window(L);
	for (int n=0; n<L; n++) {
		window[n] = 0.5*(1-cos(2*M_PI*n/(L-1)));
	}
	return window;
}




matrix linspace(double x1, double x2, unsigned int n) {
    matrix      vector(n);
    double      step = (x2-x1)/(n-1);
    vector[0]    = x1;
    vector[n-1]  = x2;

    for (int i=1; i<n-1; i++) {
        vector[i] = vector[i-1] + step;
    }

    return vector;
}

matrix diff(matrix arr) {
	matrix der(arr.sizex(), arr.sizey()-1);
	for (int j=0; j<arr.sizex(); j++) {
		for (int i=0; i<der.numel(); i++) {
			der(j, i) = arr(j, i+1) - arr(j, i);
		}
	}
	return der;
}

matrix diff(matrix arr, int amount) {
	matrix der(arr.sizex(), arr.sizey()-1);
	for (int j=0; j<arr.sizex(); j++) {
		for (int i=0; i<der.numel(); i++) {
			der(j, i) = arr(j, i+1) - arr(j, i);
		}
	}
	if (amount-1) return diff(der, amount-1);
	return der;
}
matrix conv(matrix A, matrix B) {
	int n = A.numel(), m = B.numel();
	matrix res(n + m - 1);
	int idx;
	for (int i=0; i<m+n-1; i++) {
		for (int j=i-n+1; j<=i; j++) {
			if (0<=j && j<m) {
				res[i] += A[i-j] * B[j];
			}
		}
	}
	return res;
}



double sinc(double x) {
	return x!=0 ? sin(M_PI*x)/(M_PI*x) : 1;
}

#include "mudelizer.cpp"
#endif