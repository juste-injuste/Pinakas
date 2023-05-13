#ifndef mudelizer_hpp
#define mudelizer_hpp
#include "matrix.hpp"


double err(matrix g, matrix y) {
	long double sum = 0;
	for (int i=0; i<y.sizex(); i++) {
		for (int j=0; j<y.sizey(); j++) {
			sum += g(i, j)*g(i, j) - 2*g(i, j)*y(i, j)  + y(i, j)*y(i, j);
		}
	}
	return fabs(sum);
}


double sinc(double x) {
	return x!=0 ? sin(M_PI*x)/(M_PI*x) : 1;
}

#include "mudelizer.cpp"
#endif