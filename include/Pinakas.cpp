// --inclusion guard--------------------------------------------------------------
#ifndef PINAKAS_CPP
#define PINAKAS_CPP
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  Matrix& Matrix::operator=(Matrix& B)
  {    
    const_cast<Size&>(size) = B.size;

    for (size_t index = 0; index < size.numel; ++index)
      data[0][index] = B[0][index];

    return *this;
  }
  
  inline Matrix& Matrix::operator=(Matrix&& B)
  {
    return operator=(static_cast<Matrix&>(B));
  }

  Matrix& Matrix::operator=(Value B)
  {    
    for (size_t index = 0; index < size.numel; ++index)
      data[0][index] = B;

    return *this;
  }

  Matrix operator+(Matrix& A, Matrix& B)
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
  
  inline Matrix operator+(Matrix& A, Matrix&& B)
  {
    return std::move(A + static_cast<Matrix&>(B));
  }
  
  inline Matrix operator+(Matrix&& A, Matrix& B)
  {
    return std::move(static_cast<Matrix&>(A) + B);
  }
  
  inline Matrix operator+(Matrix&& A, Matrix&& B)
  {
    return std::move(static_cast<Matrix&>(A) + static_cast<Matrix&>(B));
  }

  Matrix operator+(Matrix& A, Value B)
  {
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] += B;

    return std::move(R);
  }

  inline Matrix operator+(Matrix&& A, Value B)
  {
    return std::move(static_cast<Matrix&>(A) + B);
  }

  inline Matrix operator+(Value A, Matrix& B)
  {
    return std::move(B + A);
  }

  inline Matrix operator+(Value A, Matrix&& B)
  {
    return std::move(static_cast<Matrix&>(B) + A);
  }
  
  Matrix operator-(Matrix& A, Matrix& B)
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
  
  inline Matrix operator-(Matrix& A, Matrix&& B)
  {
    return std::move(A - static_cast<Matrix&>(B));
  }
  
  inline Matrix operator-(Matrix&& A, Matrix& B)
  {
    return std::move(static_cast<Matrix&>(A) - B);
  }
  
  inline Matrix operator-(Matrix&& A, Matrix&& B)
  {
    return std::move(static_cast<Matrix&>(A) - static_cast<Matrix&>(B));
  }
  
  Matrix operator-(Matrix& A, Value B)
  {    
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] -= B;

    return std::move(R);
  }
  
  inline Matrix operator-(Matrix&& A, Value B)
  {
    return std::move(static_cast<Matrix&>(A) - B);
  }

  inline Matrix operator-(Value A, Matrix& B)
  {
    return std::move(B - A);
  }

  inline Matrix operator-(Value A, Matrix&& B)
  {
    return std::move(static_cast<Matrix&>(B) - A);
  }
  
  Matrix operator*(Matrix& A, Matrix& B)
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
  
  inline Matrix operator*(Matrix& A, Matrix&& B)
  {
    return std::move(A * static_cast<Matrix&>(B));
  }
  
  inline Matrix operator*(Matrix&& A, Matrix& B)
  {
    return std::move(static_cast<Matrix&>(A) * B);
  }
  
  inline Matrix operator*(Matrix&& A, Matrix&& B)
  {
    return std::move(static_cast<Matrix&>(A) * static_cast<Matrix&>(B));
  }
  
  Matrix operator*(Matrix& A, Value B)
  {    
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] *= B;

    return std::move(R);
  }
  
  inline Matrix operator*(Matrix&& A, Value B)
  {
    return std::move(static_cast<Matrix&>(A) * B);
  }

  inline Matrix operator*(Value A, Matrix& B)
  {
    return std::move(B * A);
  }

  inline Matrix operator*(Value A, Matrix&& B)
  {
    return std::move(static_cast<Matrix&>(B) * A);
  }
  
  Matrix operator/(Matrix& A, Matrix& B)
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
  
  inline Matrix operator/(Matrix& A, Matrix&& B)
  {
    return std::move(A / static_cast<Matrix&>(B));
  }
  
  inline Matrix operator/(Matrix&& A, Matrix& B)
  {
    return std::move(static_cast<Matrix&>(A) / B);
  }
  
  inline Matrix operator/(Matrix&& A, Matrix&& B)
  {
    return std::move(static_cast<Matrix&>(A) / static_cast<Matrix&>(B));
  }
  
  Matrix operator/(Matrix& A, Value B)
  {
    Matrix R(A);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] /= B;

    return std::move(R);
  }
  
  inline Matrix operator/(Matrix&& A, Value B)
  {
    return std::move(static_cast<Matrix&>(A) / B);
  }

  inline Matrix operator/(Value A, Matrix& B)
  {
    Matrix R(B.size.M, B.size.N);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = A / B[0][index];

    return std::move(R);
  }

  inline Matrix operator/(Value A, Matrix&& B)
  {
    return std::move(A / static_cast<Matrix&>(B));
  }

  Matrix floor(Matrix& matrix)
  {
    Matrix R(matrix.size.M, matrix.size.N, 0);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = std::floor(matrix[0][index]);

    return std::move(R);
  }

  inline Matrix floor(Matrix&& matrix)
  {
    return std::move(floor(static_cast<Matrix&>(matrix)));
  }
  
  Matrix round(Matrix& matrix)
  {
    Matrix R(matrix.size.M, matrix.size.N, 0);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = std::round(matrix[0][index]);

    return std::move(R);
  }
  
  inline Matrix round(Matrix&& matrix)
  {
    return std::move(round(static_cast<Matrix&>(matrix)));
  }
  
  Matrix ceil(Matrix& matrix)
  {
    Matrix R(matrix.size.M, matrix.size.N, 0);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = std::ceil(matrix[0][index]);

    return std::move(R);
  }
  
  inline Matrix ceil(Matrix&& matrix)
  {
    return std::move(ceil(static_cast<Matrix&>(matrix)));
  }
  
  Matrix operator^(Matrix& A, Value B)
  {    
    Matrix R(A.size.M, A.size.N);

    for (size_t index = 0; index < R.size.numel; ++index)
      R[0][index] = std::pow(A[0][index], B);

    return std::move(R);
  }
  
  inline Matrix operator*(Matrix&& A, Value B)
  {
    return std::move(static_cast<Matrix&>(A) * B);
  }
  






  Matrix* MGS (Matrix& matrix)
  {
    size_t M = matrix.size.M, N = matrix.size.N;
    Matrix V(matrix);
    Matrix Q(M, N);
    Matrix R(N, N);
    size_t i, j, k;
    Value proj, norm;
    
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
    
    Matrix* QR = new Matrix[2]{Q, R};
    return QR;
  }
















}
// --Pinakas library: ostream overloads-------------------------------------------
namespace Pinakas::Backend
{
  std::ostream& operator<<(std::ostream& ostream, const Matrix& matrix)
  {
    // get longest number
    Value number = std::max(std::abs(matrix.min()), matrix.max());

    // find length of longest number in characters
    size_t length = std::ceil(std::log10(number)) + (matrix.min() < 0);

    // add matrix to ostream
    for (size_t y = 0; y < matrix.size.M; ++y) {
      for (size_t x = 0; x < matrix.size.N; ++x)
        ostream << std::setw(length) << matrix[y][x] << ' ';
      ostream << '\n';
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
