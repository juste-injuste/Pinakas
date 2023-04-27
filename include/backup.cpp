array array::operator | (array b) {		// solve linear system using QR decomposition and back-substitution
	int 	N 		= size.x;

	array u[N] 		= array(1, size.y);
	array v[N] 		= array(1, size.y);
	array Q(size.x, size.y);
	array R(size.x, size.x);
	double norm[N];
	double temp;

	for (int i=0; i<N; i++) {			// compute Q and R using Gram-Schmidt process
		v[i] = get_col(i);
		u[i] = v[i];
		for (int j=0; j<i; j++) {
			temp = inner(v[i], u[j]);
			u[i] -= u[j] * temp;
			R(i, j) = temp;
		}
		norm[i] 	= sqrt(inner(u[i], u[i]));
		u[i] 	   /= norm[i];
		R(i, i)		= norm[i];
		for (int j=0; j<size.y; j++)
			Q(i, j) = u[i].value[j];
	}

	array Qtb(b.size.x, Q.size.x);		// compute Q' * b
	long double sum;
	for (int i=0; i<b.size.x; i++) {
		for (int j=0; j<Q.size.x; j++) {
			sum = 0;
			for (int r=0; r<Q.size.y; r++)
				sum += Q(j, r) * b(i, r);
			Qtb(i, j) = sum;
		}
	}
	
	array x(1, size.x);					// solve Rx = Qtb using back-substitution
	for (int i=N-1; i>=0; i--) {
		x.value[i] = Qtb[i];
		for (int j=N-1; j>i; j--)
			x.value[i] -= R(j, i) * x.value[j];
		x.value[i] /= R(i, i);
	}
	
	return x;
}

array array::operator | (array b) {		// solve linear system using QR decomposition and back-substitution
	int   N = size.x, M = size.y;
	double Q[N][M];
	double R[N][N];
	double norm[N];
	double proj;
	int i, j, k;
	for (i=0; i<N; i++) {			// compute Q and R using Gram-Schmidt process
		for (j=0; j<M; j++) {
			Q[i][j] = value[i + j*N];
		}
		for (j=0; j<i; j++) {
			proj = 0;
			for (k=0; k<M; k++) {
				proj += value[i + k*N] * Q[j][k];
			} // <v[i], u[j])>
			for (k=0; k<M; k++) {
				Q[i][k] -= Q[j][k] * proj;
			} // u[i] -= u[j] * <v[i], u[j])>;
			R[i][j] = proj;
		}
		proj = 0;
		for (j=0; j<M; j++) {
			proj += Q[i][j] * Q[i][j];
		} // <u[i], u[j])>
		norm[i] = sqrt(proj);
		for (j=0; j<M; j++) {
			Q[i][j] /= norm[i];
		} // u[i] /= u[i]/|u[i]|;
		R[i][i]	= norm[i];
	}

	double Qtb[N];		// compute Q' * b
	long double sum;
	for (i=0; i<N; i++) {
		sum = 0;
		for (j=0; j<M; j++)
			sum += Q[i][j] * b.value[j];
		Qtb[i] = sum;
	}
	
	array x(1, N);		// solve Rx = Qtb using back-substitution
	for (i=N-1; i>=0; i--) {
		x.value[i] = Qtb[i];
		for (j=N-1; j>i; j--)
			x.value[i] -= R[j][i] * x.value[j];
		x.value[i] /= R[i][i];
	}
	
	return x;
}
