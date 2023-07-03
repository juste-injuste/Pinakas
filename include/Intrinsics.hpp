#ifndef INTRINSICS_HPP
#define INTRINSICS_HPP
#include "Pinakas.hpp"

#include <immintrin.h>
namespace Pinakas { namespace Backend
{


Matrix<double>& add_mat_inplace_fast(Matrix<double>& A, const Matrix<double>& B)
{
  if (A.size() != B.size()) {
    std::cerr << "error: add_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
    return A;
  }

  const size_t n = A.numel();

  // Determine the number of elements that can be processed simultaneously using AVX-512
  constexpr size_t SimdSize = sizeof(__m512d) / sizeof(double); // Assuming AVX-512 instructions are available

  // Process elements in parallel using AVX-512 intrinsics
  for (size_t k = 0; k < n; k += SimdSize) {
    // Load SimdSize elements from A and B
    __m512d aVec = _mm512_loadu_pd(&A[0][k]);
    __m512d bVec = _mm512_loadu_pd(&B[0][k]);

    // Perform element-wise addition using AVX-512
    __m512d resultVec = _mm512_add_pd(aVec, bVec);

    // Store the result back to A
    _mm512_storeu_pd(&A[0][k], resultVec);
  }

  // Handle any remaining elements that couldn't be processed in parallel
  for (size_t k = n - (n % SimdSize); k < n; ++k)
    A[0][k] += B[0][k];

  return A;
}


Matrix<double> div_fast(const Matrix<double>& b, Matrix<double> A)
{
  // verify vertical dimensions
  if (b.M() != A.M()) {
    std::cerr << "error: div: vertical dimensions mismatch (b is " << b.M() << "x_, A is " << A.M() << "x_)\n";
    return Matrix<double>();
  }

  // verify that b is a column matrix
  if (b.N() != 1) {
    std::cerr << "error: div: b's horizontal dimension is not 1 (b is _x" << b.N() << ")\n";
    return Matrix<double>();
  }

  // store the dimensions of A
  const size_t M = A.M();
  const size_t N = A.N();

  // QR decomposition matrices and result matrix
  Matrix<double> Q(M, N), R(N, N), x(N, 1);

  // QR decomposition using the modified Gram-Schmidt process
  for (size_t i = 0; i < N; ++i) {
    // calculate the squared Euclidean norm of A's i'th column
    double sum_of_squares = 0;
    for (size_t j = 0; j < M; ++j)
      sum_of_squares += A[j][i] * A[j][i];

    // skips if the squared Euclidean norm is 0
    if (sum_of_squares != 0) {
      // calculate the inverse Euclidean norm of A's i'th column
      __m512d inorm = _mm512_set1_pd(std::pow(sum_of_squares, -0.5));
      // normalize and store A's normalized i'th column using AVX-512
      for (size_t j = 0; j < M; j += 8) {
        __m512d A_col = _mm512_loadu_pd(&A[j][i]);
        __m512d Q_col = _mm512_mul_pd(A_col, inorm);
        _mm512_storeu_pd(&Q[j][i], Q_col);
      }
    }

    // orthogonalize the remaining columns with respect to A's i'th column
    for (size_t k = i; k < N; ++k) {
      // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column using AVX-512
      __m512d projection = _mm512_setzero_pd();
      for (size_t j = 0; j < M; j += 8) {
        __m512d Q_col = _mm512_loadu_pd(&Q[j][i]);
        __m512d A_col = _mm512_loadu_pd(&A[j][k]);
        projection = _mm512_fmadd_pd(Q_col, A_col, projection);
      }

      // construct upper triangle matrix R using Q's i'th orthonormal projection onto A's k'th unorthogonalized column
      if (k >= i) {
        double R_val[8];
        _mm512_storeu_pd(R_val, projection);
        for (size_t l = 0; l < 8 && i + l < N; ++l)
          R[i][k + l] = R_val[l];
      }

      // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
      if (k != i) { // skips if k == i because the projection would be 0
        for (size_t j = 0; j < M; j += 8) {
          __m512d projection = _mm512_loadu_pd(Q.get() + j + i * M);
          __m512d A_col = _mm512_loadu_pd(A.get() + j + k * M);
          A_col = _mm512_fnmadd_pd(projection, A_col, A_col);
          _mm512_storeu_pd(A.get() + j + k * M, A_col);
        }
      }
    }
  }

    // solve linear system Rx = Qt*b using back substitution
    for (size_t i = N - 1; i < N; --i) {
      // calculate appropriate Qt*b component
      double substitution = 0;
      for (size_t j = 0; j < M; ++j)
        substitution += Q[j][i] * b[j][0];

      // back substitution of previously solved x components
      for (size_t k = N - 1; k > i; --k)
        substitution -= R[i][k] * x[k][0];

      // solve x's i'th component
      x[i][0] = substitution / R[i][i];
    }
    
    return x;
}











}}


#endif