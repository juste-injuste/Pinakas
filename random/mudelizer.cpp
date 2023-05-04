#include "mudelizer.hpp"


matrix resample(matrix data, unsigned int L) {
	int N	= data.numel()*L;	// approximate new data lenght
	int o	= 3 * L;			// offset for filter impulse
	int l	= 2 * o + 1;		// filter impulse lenght

	double u[N] = {0};	// upsampled data
	for (int i=0; i<data.numel(); i++) {
		u[i*L] = data[i];
	}

	double h, w, hw[l];			// windowed filter impulse
	for (int n=0; n<l; n++) {
		h 		= (n-o) ? sin((n-o)*M_PI/L)/((n-o)*M_PI/L) : 1;
		w 		= 0.42 - 0.5*cos(2*M_PI*n/(l-1)) + 0.08*cos(4*M_PI*n/(l-1));
		hw[n] 	= h*w;
	}

	matrix r(N-L+1);		// cropped convolution of upsampled data with windowed filter impulse
	for (int i=0; i<N-L+1; i++) {
		for (int n=0; n<l; n++) {
			r[i] += u[(i+n-o) < N ? abs(i+n-o) : 2*N-L - (i+n-o)] * hw[n];
		}
	}

	return r;
}



matrix fir_filter(matrix data, matrix impulse) {
	int N	= data.numel();		// approximate new data lenght
	int l	= impulse.numel();	// filter impulse lenght
	int o	= (l-1) >> 1;		// offset for filter impulse
	int L	= o/3;

	matrix f(N-L+1);		// cropped convolution of upsampled data with windowed filter impulse
	for (int i=0; i<N-L+1; i++) {
		for (int n=0; n<l; n++) {
			f[i] += data[(i+n-o) < N ? abs(i+n-o) : 2*N-L - (i+n-o)] * impulse[n];
		}
	}

	return f;
}



matrix upsample(matrix data, unsigned int L) {
	int N	= data.numel()*L;	// approximate new data lenght
	int o	= 3 * L;			// offset for filter impulse
	int l	= 2 * o + 1;		// filter impulse lenght

	matrix u(N);				// upsampled data
	for (int i=0; i<data.numel(); i++) {
		u[i*L] = data[i];
	}

	return u;
}


matrix design_filter(unsigned int l) {
	int o	= (l-1) >> 1;	// offset for filter impulse

	matrix h(l);			// windowed filter impulse
	for (int n=0; n<l; n++) {
		h[n] 	= (n-o) ? sin((n-o)*3*M_PI/o)/((n-o)*3*M_PI/o) : 1;
	}

	return h;
}


