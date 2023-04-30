#ifndef mudelizer_hpp
#define mudelizer_hpp
#include "matrix.hpp"


matrix knee(matrix arr, double n);


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
		root -= tol*f(root)/(f(root+tol_2)-f(root-tol_2));
		//cout << i << endl;
	}

	return root;
}



double sinc(double x) {
	return x!=0 ? sin(M_PI*x)/(M_PI*x) : 1;
}

#include "mudelizer.cpp"
#endif