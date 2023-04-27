// --inclusion guard--------------------------------------------------------------
#ifndef PINAKAS_CPP
#define PINAKAS_CPP
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  Matrix& Matrix::operator=(const Matrix& B)
  {    
    const_cast<Size&>(size) = B.size;

    for (size_t index = 0; index < size.numel; ++index)
      data[0][index] = B[0][index];

    return *this;
  }
  
  inline Matrix& Matrix::operator=(const Matrix&& B)
  {
    return operator=(const_cast<Matrix&>(B));
  }

  Matrix& Matrix::operator=(const Value B)
  {    
    for (size_t index = 0; index < size.numel; ++index)
      data[0][index] = B;

    return *this;
  }

  Matrix operator+(const Matrix& A, const Matrix& B)
  {
    if ((A.size.M != B.size.M) || (A.size.N != B.size.N)) {
      std::cerr << "operator +: nonconformant arguments (A is " << A.size << ", B is " << B.size << ")\n";
      return A;
    }
    
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] += B[0][index];

    return std::move(R);
  }
  
  inline Matrix operator+(const Matrix& A, const Matrix&& B)
  {
    return std::move(A + const_cast<Matrix&>(B));
  }
  
  inline Matrix operator+(const Matrix&& A, const Matrix& B)
  {
    return std::move(const_cast<Matrix&>(A) + B);
  }
  
  inline Matrix operator+(const Matrix&& A, const Matrix&& B)
  {
    return std::move(const_cast<Matrix&>(A) + const_cast<Matrix&>(B));
  }

  Matrix operator+(const Matrix& A, const Value B)
  {
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] += B;

    return std::move(R);
  }

  inline Matrix operator+(const Matrix&& A, const Value B)
  {
    return std::move(const_cast<Matrix&>(A) + B);
  }

  inline Matrix operator+(const Value A, const Matrix& B)
  {
    return std::move(B + A);
  }

  inline Matrix operator+(const Value A, const Matrix&& B)
  {
    return std::move(const_cast<Matrix&>(B) + A);
  }
  
  Matrix operator-(const Matrix& A, const Matrix& B)
  {
    if ((A.size.M != B.size.M) || (A.size.N != B.size.N)) {
      std::cerr << "operator -: nonconformant arguments (A is " << A.size << ", B is " << B.size << ")\n";
      return A;
    }
    
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] -= B[0][index];

    return std::move(R);
  }
  
  inline Matrix operator-(const Matrix& A, const Matrix&& B)
  {
    return std::move(A - const_cast<Matrix&>(B));
  }
  
  inline Matrix operator-(const Matrix&& A, const Matrix& B)
  {
    return std::move(const_cast<Matrix&>(A) - B);
  }
  
  inline Matrix operator-(const Matrix&& A, const Matrix&& B)
  {
    return std::move(const_cast<Matrix&>(A) - const_cast<Matrix&>(B));
  }
  
  Matrix operator-(const Matrix& A, const Value B)
  {    
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] -= B;

    return std::move(R);
  }
  
  inline Matrix operator-(const Matrix&& A, const Value B)
  {
    return std::move(const_cast<Matrix&>(A) - B);
  }

  inline Matrix operator-(const Value A, const Matrix& B)
  {
    return std::move(B - A);
  }

  inline Matrix operator-(const Value A, const Matrix&& B)
  {
    return std::move(const_cast<Matrix&>(B) - A);
  }
  
  Matrix operator*(const Matrix& A, const Matrix& B)
  {
    if ((A.size.M != B.size.M) || (A.size.N != B.size.N)) {
      std::cerr << "operator *: nonconformant arguments (A is " << A.size << ", B is " << B.size << ")\n";
      return A;
    }
    
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] *= B[0][index];

    return std::move(R);
  }
  
  inline Matrix operator*(const Matrix& A, const Matrix&& B)
  {
    return std::move(A * const_cast<Matrix&>(B));
  }
  
  inline Matrix operator*(const Matrix&& A, const Matrix& B)
  {
    return std::move(const_cast<Matrix&>(A) * B);
  }
  
  inline Matrix operator*(const Matrix&& A, const Matrix&& B)
  {
    return std::move(const_cast<Matrix&>(A) * const_cast<Matrix&>(B));
  }
  
  Matrix operator*(const Matrix& A, const Value B)
  {    
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] *= B;

    return std::move(R);
  }
  
  inline Matrix operator*(const Matrix&& A, const Value B)
  {
    return std::move(const_cast<Matrix&>(A) * B);
  }

  inline Matrix operator*(const Value A, const Matrix& B)
  {
    return std::move(B * A);
  }

  inline Matrix operator*(const Value A, const Matrix&& B)
  {
    return std::move(const_cast<Matrix&>(B) * A);
  }
  
  Matrix operator/(const Matrix& A, const Matrix& B)
  {
    if ((A.size.M != B.size.M) || (A.size.N != B.size.N)) {
      std::cerr << "operator /: nonconformant arguments (A is " << A.size << ", B is " << B.size << ")\n";
      return A;
    }
    
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] /= B[0][index];

    return std::move(R);
  }
  
  inline Matrix operator/(const Matrix& A, const Matrix&& B)
  {
    return std::move(A / const_cast<Matrix&>(B));
  }
  
  inline Matrix operator/(const Matrix&& A, const Matrix& B)
  {
    return std::move(const_cast<Matrix&>(A) / B);
  }
  
  inline Matrix operator/(const Matrix&& A, const Matrix&& B)
  {
    return std::move(const_cast<Matrix&>(A) / const_cast<Matrix&>(B));
  }
  
  Matrix operator/(const Matrix& A, const Value B)
  {
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] /= B;

    return std::move(R);
  }
  
  inline Matrix operator/(const Matrix&& A, const Value B)
  {
    return std::move(const_cast<Matrix&>(A) / B);
  }

  inline Matrix operator/(const Value A, const Matrix& B)
  {
    Matrix R(B.size.M, B.size.N);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = A / B[0][index];

    return std::move(R);
  }

  inline Matrix operator/(const Value A, const Matrix&& B)
  {
    return std::move(A / const_cast<Matrix&>(B));
  }

  Matrix floor(const Matrix& A)
  {
    Matrix R(A.size.M, A.size.N, 0);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = std::floor(A[0][index]);

    return std::move(R);
  }

  inline Matrix floor(const Matrix&& A)
  {
    return std::move(floor(const_cast<Matrix&>(A)));
  }
  
  Matrix round(const Matrix& A)
  {
    Matrix R(A.size.M, A.size.N, 0);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = std::round(A[0][index]);

    return std::move(R);
  }
  
  inline Matrix round(const Matrix&& A)
  {
    return std::move(round(const_cast<Matrix&>(A)));
  }
  
  Matrix ceil(Matrix& A)
  {
    Matrix R(A.size.M, A.size.N, 0);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = std::ceil(A[0][index]);

    return std::move(R);
  }
  
  inline Matrix ceil(const Matrix&& A)
  {
    return std::move(ceil(const_cast<Matrix&>(A)));
  }
  
  Matrix operator^(const Matrix& A, const Value B)
  {    
    Matrix R(A.size.M, A.size.N);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = std::pow(A[0][index], B);

    return std::move(R);
  }
  
  inline Matrix operator^(const Matrix&& A, const Value B)
  {
    return std::move(const_cast<Matrix&>(A) ^ B);
  }
  

  Matrix mul(const Matrix& A, const Matrix& B)
  {
    Matrix R(A.size.M, B.size.N, 0);
    for (size_t i = 0; i<B.size.N; i++)
      for (size_t j = 0; j<A.size.M; j++)
        for (size_t k = 0; k<A.size.N; k++)
          R[j][i] += A[j][k] * B[k][i];
    return std::move(R);
  }
  
  inline Matrix mul(const Matrix& A, const Matrix&& B)
  {
    return std::move(mul(A, const_cast<Matrix&>(B)));
  }
  
  inline Matrix mul(const Matrix&& A, const Matrix& B)
  {
    return std::move(mul(const_cast<Matrix&>(A), B));
  }
  
  inline Matrix mul(const Matrix&& A, const Matrix&& B)
  {
    return std::move(mul(const_cast<Matrix&>(A), const_cast<Matrix&>(B)));
  }

  Matrix transpose(const Matrix& A)
  {
    Matrix R(A.size.N, A.size.M);
    for (size_t y = 0; y < A.size.M; ++y)
      for (size_t x = 0; x < A.size.N; ++x)
        R[x][y] = A[y][x];
    return std::move(R);
  }

  inline Matrix transpose(const Matrix&& A)
  {
    return std::move(const_cast<Matrix&>(A));
  }

  std::unique_ptr<Matrix[]> MGS(const Matrix& A)
  {
    size_t i, j, k;
    size_t M = A.size.M;
    size_t N = A.size.N;
    Matrix V(A);
    Matrix* QR = new Matrix[2]{Matrix(M, N, 0), Matrix(N, N, 0)};
    Value sum_of_squares, norm, projection;

    for (j = 0; j < N; ++j) {
      // q_j = v_j / ||v_j||_2
      for (i = 0, sum_of_squares = 0; i < M; ++i)
        sum_of_squares += V[i][j] * V[i][j];
      norm = std::sqrt(sum_of_squares);
      if (norm) for (i = 0; i < M; ++i)
        QR[0][i][j] += V[i][j] / norm;
      for (k = j; k < N; ++k) {
        // v_k = v_k - (qT_j*v_k)*q_j
        for (i = 0, projection = 0; i < M; ++i)
          projection += QR[0][i][j] * V[i][k];
        for (i = 0; i < M; ++i)
          V[i][k] -= projection * QR[0][i][j];
        // compute R
        if (k >= j)
          QR[1][j][k] = projection;
      }
    }
    return std::unique_ptr<Matrix[]>(QR);
  }

  inline std::unique_ptr<Matrix[]> MGS(const Matrix&& A)
  {
    return MGS(const_cast<Matrix&>(A));
  }
}
// --Pinakas library: ostream overloads-------------------------------------------
namespace Pinakas::Backend
{
  std::ostream& operator<<(std::ostream& ostream, const Matrix& A)
  {
    if (A.size.numel) {
      // get longest number
      Value number = std::max(std::abs(max(A)), max(A));

      // find length of longest number in characters
      size_t length = std::ceil(std::log10(number)) + (max(A) < 0);

      // add matrix to ostream
      for (size_t y = 0; y < A.size.M; ++y) {
        for (size_t x = 0; x < A.size.N; ++x)
          ostream << std::setw(length) << std::left << A[y][x] << ' ';
        ostream << '\n';
      }
    }
    return ostream;
  }

  inline std::ostream& operator<<(std::ostream& ostream, const Matrix::Size size)
  {
    return ostream << size.N << 'x' << size.M;
  }
  
}
// --Pinakas library: include definitions-----------------------------------------
#include "Pinakas.cpp"
#endif
