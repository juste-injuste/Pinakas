#ifndef INTRINSICS_HPP
#define INTRINSICS_HPP
#include "Pinakas.hpp"

#include <immintrin.h>

namespace Pinakas
{
  namespace Backend
  {
    #if defined(__GNUC__)
      #define PINAKAS_INLINE __attribute__((always_inline)) inline
    #elif defined(__clang__)
      #define PINAKAS_INLINE __attribute__((always_inline)) inline
    #elif defined(__apple_build_version__)
      #define PINAKAS_INLINE __attribute__((always_inline)) inline
    #elif defined(__MINGW32__)
      #define PINAKAS_INLINE __attribute__((always_inline)) inline
    #elif defined(__MINGW64__)
      #define PINAKAS_INLINE __attribute__((always_inline)) inline
    #elif defined(_MSC_VER)
      #define PINAKAS_INLINE __forceinline
    #elif defined(__INTEL_COMPILER)
      #define PINAKAS_INLINE __forceinline
    #elif defined(__SUNPRO_C)
      #define PINAKAS_INLINE __forceinline
    #elif defined(__SUNPRO_CC)
      #define PINAKAS_INLINE __forceinline
    #elif defined(__ARMCC_VERSION)
      #define PINAKAS_INLINE __forceinline
    #elif defined(__IBMC__)
      #define PINAKAS_INLINE __inline__
    #elif defined(__xlC__)
      #define PINAKAS_INLINE __inline__
    #else
      #define PINAKAS_INLINE inline
    #endif

    #ifdef __AVX512F__
      #define PINAKAS_USE_AVX512 
      #define PINAKAS_USE_PARALLELISM
      const char* instruction_set = "AVX512";
      
      PINAKAS_INLINE __m512d simd_load(double* ptr)
      {
        return _mm512_loadu_pd(ptr);
      }

      PINAKAS_INLINE __m512 simd_load(float* ptr)
      {
        return _mm512_loadu_ps(ptr);
      }
      
      PINAKAS_INLINE void simd_store(double* ptr, __m512d a)
      {
        _mm512_storeu_pd(ptr, a);
      }

      PINAKAS_INLINE void simd_store(float* ptr, __m512 a)
      {
        _mm512_storeu_ps(ptr, a);
      }

      PINAKAS_INLINE __m512d simd_set1(double value)
      {
        return _mm512_set1_pd(value);
      }

      PINAKAS_INLINE __m512 simd_set1(float value)
      {
        return _mm512_set1_ps(value);
      }

      template<typename T>
      PINAKAS_INLINE T simd_setzero();

      template<>
      PINAKAS_INLINE __m512d simd_setzero()
      {
        return _mm512_setzero_pd();
      }

      template<>
      PINAKAS_INLINE __m512 simd_setzero()
      {
        return _mm512_setzero_ps();
      }

      PINAKAS_INLINE __m512d simd_mul(__m512d a, __m512d b)
      {
        return _mm512_mul_pd(a, b);
      }

      PINAKAS_INLINE __m512 simd_mul(__m512 a, __m512 b)
      {
        return _mm512_mul_ps(a, b);
      }

      PINAKAS_INLINE __m512d simd_muladd(__m512d a, __m512d b, __m512d c)
      {
        return _mm512_fmadd_pd(a, b, c);
      }

      PINAKAS_INLINE __m512 simd_muladd(__m512 a, __m512 b, __m512 c)
      {
        return _mm512_fmadd_ps(a, b, c);
      }

      PINAKAS_INLINE __m512d simd_nmuladd(__m512d a, __m512d b, __m512d c)
      {
        return _mm512_fnmadd_pd(a, b, c);
      }

      PINAKAS_INLINE __m512 simd_nmuladd(__m512 a, __m512 b, __m512 c)
      {
        return _mm512_fnmadd_ps(a, b, c);
      }

      __m512d get_simd_type(double)
      {
        return __m512d();
      }

      __m512 get_simd_type(float)
      {
        return __m512();
      }
    #elif defined(__AVX2__) || defined(__AVX__)
      #ifdef __AVX2__
        #define PINAKAS_USE_AVX2
        const char* instruction_set = "AVX2";
      #else
        #define PINAKAS_USE_AVX
        const char* instruction_set = "AVX";
      #endif
      #define PINAKAS_USE_PARALLELISM
      
      PINAKAS_INLINE __m256d simd_load(double* ptr)
      {
        return _mm256_loadu_pd(ptr);
      }

      PINAKAS_INLINE __m256 simd_load(float* ptr)
      {
        return _mm256_loadu_ps(ptr);
      }
      
      PINAKAS_INLINE void simd_store(double* ptr, __m256d a)
      {
        _mm256_storeu_pd(ptr, a);
      }

      PINAKAS_INLINE void simd_store(float* ptr, __m256 a)
      {
        _mm256_storeu_ps(ptr, a);
      }

      PINAKAS_INLINE __m256d simd_set1(double value)
      {
        return _mm256_set1_pd(value);
      }

      PINAKAS_INLINE __m256 simd_set1(float value)
      {
        return _mm256_set1_ps(value);
      }

      template<typename T>
      PINAKAS_INLINE T simd_setzero();

      template<>
      PINAKAS_INLINE __m256d simd_setzero()
      {
        return _mm256_setzero_pd();
      }

      template<>
      PINAKAS_INLINE __m256 simd_setzero()
      {
        return _mm256_setzero_ps();
      }

      PINAKAS_INLINE __m256d simd_mul(__m256d a, __m256d b)
      {
        return _mm256_mul_pd(a, b);
      }

      PINAKAS_INLINE __m256 simd_mul(__m256 a, __m256 b)
      {
        return _mm256_mul_ps(a, b);
      }

      PINAKAS_INLINE __m256d simd_muladd(__m256d a, __m256d b, __m256d c)
      {
        return _mm256_fmadd_pd(a, b, c);
      }

      PINAKAS_INLINE __m256 simd_muladd(__m256 a, __m256 b, __m256 c)
      {
        return _mm256_fmadd_ps(a, b, c);
      }

      PINAKAS_INLINE __m256d simd_nmuladd(__m256d a, __m256d b, __m256d c)
      {
        return _mm256_fnmadd_pd(a, b, c);
      }

      PINAKAS_INLINE __m256 simd_nmuladd(__m256 a, __m256 b, __m256 c)
      {
        return _mm256_fnmadd_ps(a, b, c);
      }

      __m256d get_simd_type(double)
      {
        return __m256d();
      }

      __m256 get_simd_type(float)
      {
        return __m256();
      }
    #elif defined(__SSE__)
      #define PINAKAS_USE_SSE
      #define PINAKAS_USE_PARALLELISM
      const char* instruction_set = "SSE";
      PINAKAS_INLINE __m128d simd_load(double* ptr)
      {
        return _mm_loadu_pd(ptr);
      }

      PINAKAS_INLINE __m128 simd_load(float* ptr)
      {
        return _mm_loadu_ps(ptr);
      }
      
      PINAKAS_INLINE void simd_store(double* ptr, __m128d a)
      {
        _mm_storeu_pd(ptr, a);
      }

      PINAKAS_INLINE void simd_store(float* ptr, __m128 a)
      {
        _mm_storeu_ps(ptr, a);
      }

      PINAKAS_INLINE __m128d simd_set1(double value)
      {
        return _mm_set1_pd(value);
      }

      PINAKAS_INLINE __m128 simd_set1(float value)
      {
        return _mm_set1_ps(value);
      }

      template<typename T>
      PINAKAS_INLINE T simd_setzero();

      template<>
      PINAKAS_INLINE __m128d simd_setzero()
      {
        return _mm_setzero_pd();
      }

      template<>
      PINAKAS_INLINE __m128 simd_setzero()
      {
        return _mm_setzero_ps();
      }

      PINAKAS_INLINE __m128d simd_mul(__m128d a, __m128d b)
      {
        return _mm_mul_pd(a, b);
      }

      PINAKAS_INLINE __m128 simd_mul(__m128 a, __m128 b)
      {
        return _mm_mul_ps(a, b);
      }

      PINAKAS_INLINE __m128d simd_muladd(__m128d a, __m128d b, __m128d c)
      {
        return _mm_add_pd(_mm_mul_pd(a, b), c);
      }

      PINAKAS_INLINE __m128 simd_muladd(__m128 a, __m128 b, __m128 c)
      {
        return _mm_add_ps(_mm_mul_ps(a, b), c);
      }

      PINAKAS_INLINE __m128d simd_nmuladd(__m128d a, __m128d b, __m128d c)
      {
        return _mm_sub_pd(c, _mm_mul_pd(a, b));
      }

      PINAKAS_INLINE __m128 simd_nmuladd(__m128 a, __m128 b, __m128 c)
      {
        return _mm_sub_ps(c, _mm_mul_ps(a, b));
      }

      __m128d get_simd_type(double)
      {
        return __m128d();
      }

      __m128 get_simd_type(float)
      {
        return __m128();
      }
    #else
      #define PINAKAS_USE_NO_PARALLELISM
      const char* instruction_set = "no SIMD set";
    #endif

#ifdef PINAKAS_USE_PARALLELISM
    template<typename T, typename T0 = convert_to_double<T>>
    Matrix<T> div_fast(const Matrix<T>& b, Matrix<T> A)
    {
      using simd_type = decltype(get_simd_type(T()));
      constexpr size_t simd_size = sizeof(simd_type) / sizeof(T);

      // verify vertical dimensions
      if (b.M() != A.M()) {
        std::cerr << "error: div: vertical dimensions mismatch (b is " << b.M() << "x_, A is " << A.M() << "x_)\n";
        return Matrix<T>();
      }

      // verify that b is a column matrix
      if (b.N() != 1) {
        std::cerr << "error: div: b's horizontal dimension is not 1 (b is _x" << b.N() << ")\n";
        return Matrix<T>();
      }

      // store the dimensions of A
      const size_t M = A.M();
      const size_t N = A.N();

      // QR decomposition matrices and result matrix
      Matrix<T> Q(M, N), R(N, N), x(N, 1);

      // QR decomposition using the modified Gram-Schmidt process
      for (size_t i = 0; i < N; ++i) {
        // calculate the squared Euclidean norm of A's i'th column
        double sum_of_squares = 0;
        for (size_t j = 0; j < M; ++j)
          sum_of_squares += A[j][i] * A[j][i];

        // skips if the squared Euclidean norm is 0
        if (sum_of_squares != 0) {
          // calculate the inverse Euclidean norm of A's i'th column
          simd_type inorm = simd_set1(T(std::pow(sum_of_squares, -0.5)));
          // normalize and store A's normalized i'th column using AVX-512
          for (size_t j = 0; j < M; j += simd_size) {
            simd_type A_col = simd_load(&A[j][i]);
            simd_type Q_col = simd_mul(A_col, inorm);
            simd_store(&Q[j][i], Q_col);
          }
        }

        // orthogonalize the remaining columns with respect to A's i'th column
        for (size_t k = i; k < N; ++k) {
          // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column using AVX-512
          simd_type projection = simd_setzero<simd_type>();
          for (size_t j = 0; j < M; j += simd_size) {
            simd_type Q_col = simd_load(&Q[j][i]);
            simd_type A_col = simd_load(&A[j][k]);
            projection = simd_muladd(Q_col, A_col, projection);
          }

          // construct upper triangle matrix R using Q's i'th orthonormal projection onto A's k'th unorthogonalized column
          if (k >= i) {
            T R_val[simd_size];
            simd_store(R_val, projection);
            for (size_t l = 0; l < simd_size && i + l < N; ++l)
              R[i][k + l] = R_val[l];
          }

          // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
          if (k != i) { // skips if k == i because the projection would be 0
            for (size_t j = 0; j < M; j += simd_size) {
              simd_type projection = simd_load(&Q[j][k]);
              simd_type A_col = simd_load(&A[j][k]);
              A_col = simd_nmuladd(projection, A_col, A_col);
              simd_store(&A[j][k], A_col);
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
#endif




  }
}
#endif