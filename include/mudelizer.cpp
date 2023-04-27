#include "mudelizer.hpp"

matrix_array MGS(matrix arr) {
	int   N = arr.sizex(), M = arr.sizey();
	double 	V[N][M];
	matrix 	Q(N, M);
	matrix 	R(N, N);
	register int i, j, k;
	double proj, norm;
	
	for (j=0; j<N; j++)
		for (i=0; i<M; i++)
			V[j][i] = arr(j, i);
	for (j=0; j<N; j++) {
		proj = 0;
		for (i=0; i<M; i++) {
			proj += V[j][i] * V[j][i];
		}
		norm = sqrt(proj);
		R(j, j) = norm;
		for (i=0; i<M; i++) {
			Q(j, i) = V[j][i] / norm;
		}
		for (k=j+1; k<N; k++) {
			proj = 0;
			for (i=0; i<M; i++) {
				proj += Q(j, i) * V[k][i];
			}
			for (i=0; i<M; i++) {
				V[k][i] -= Q(j, i) * proj;
			}
			R(k, j) = proj;
		}
	}
	matrix_array QR(2);
	QR[0] = Q;
	QR[1] = R;
	return QR;
}

matrix_array linearize(matrix xdata, matrix ydata) {  // resample un-equally spaced sorted data
    int     n = xdata.numel();
    matrix  new_x(n), new_y(n);
    double  step = (xdata[n-1]-xdata[0])/(n-1);
	double  x1, x2, y1, y2;

    new_x[0]	= xdata[0];
	new_y[0]    = ydata[0];
    new_x[n-1]  = xdata[n-1];
    new_y[n-1]  = ydata[n-1];

    for (int i=1; i<n-1; i++) {
        new_x[i] = new_x[i-1] + step;

        x1 = xdata[i];
		y1 = ydata[i];
        x2 = xdata[i+1];
        y2 = ydata[i+1];

        new_y[i] = ((y1-y2)*new_x[i] + x1*y2 - x2*y1)/(x1-x2);
    }

	matrix_array resampled(2);
	resampled[0] = new_x;
	resampled[1] = new_y;
    return resampled;
}

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


