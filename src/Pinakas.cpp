// --inclusion guard--------------------------------------------------------------
#include "../include/Pinakas.hpp"
#define M_PI 3.14159265358979323846
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  Matrix& Matrix::operator=(const Matrix& B) noexcept
  {
    std::clog << "Matrix assigned !\n";
    if (_size != B._size) allocate(B._size.M, B._size.N);

    if (_memory_block.get()) {
      for (size_t index = 0; index < _size.numel; ++index)
        _data[0][index] = B[0][index];
    }

    return *this;
  }
  
  Matrix& Matrix::operator=(Matrix&& B) noexcept
  {
    std::clog << "Matrix move-assigned !\n";
    _size = B._size;
    _memory_block.reset(B._memory_block.release());
    _data = B._data;
    return *this;
  }

  Matrix& Matrix::operator=(const Value B) noexcept
  {    
    for (size_t index = 0; index < _size.numel; ++index)
      _data[0][index] = B;

    return *this;
  }
// -------------------------------------------------------------------------------
  Matrix& operator+=(Matrix& A, const Matrix& B)
  {
    if ((A.size().M != B.size().M) || (A.size().N != B.size().N)) {
      std::cerr << "operator +=: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return A;
    }

    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += B[0][index];

    return A;
  }

  Matrix& operator+=(Matrix& A, const Value B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += B;

    return A;
  }
// -------------------------------------------------------------------------------
  Matrix operator+(const Matrix& A, const Matrix& B)
  {
    if ((A.size().M != B.size().M) || (A.size().N != B.size().N)) {
      std::cerr << "operator +: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return A;
    }
    
    Matrix R(A);

    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] += B[0][index];

    return R;
  }

  inline Matrix&& operator+(Matrix&& A, Matrix&& B)
  {
    return std::move(A += B);
  }

  Matrix operator+(const Matrix& A, const Value B)
  {
    Matrix R(A);

    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] += B;

    return R;
  }

  inline Matrix&& operator+(Matrix&& A, const Value B)
  {
    return std::move(A += B);
  }

  inline Matrix operator+(const Value A, const Matrix& B)
  {
    return B + A;
  }

  inline Matrix&& operator+(const Value A, Matrix&& B)
  {
    return std::move(B += A);
  }

  Matrix floor(const Matrix& A)
  {
    Matrix R(A.size().M, A.size().N, 0);

    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = std::floor(A[0][index]);

    return R;
  }

  inline Matrix floor(const Matrix&& A)
  {
    return floor(const_cast<Matrix&>(A));
  }
  
  Matrix round(const Matrix& A)
  {
    Matrix R(A.size().M, A.size().N, 0);

    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = std::round(A[0][index]);

    return R;
  }
  
  inline Matrix round(const Matrix&& A)
  {
    return round(const_cast<Matrix&>(A));
  }
  
  Matrix ceil(Matrix& A)
  {
    Matrix R(A.size(), 0);

    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = std::ceil(A[0][index]);

    return R;
  }
  
  inline Matrix ceil(const Matrix&& A)
  {
    return ceil(const_cast<Matrix&>(A));
  }
  
  Matrix operator^(const Matrix& A, const Value B)
  {    
    Matrix R(A.size());

    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = std::pow(A[0][index], B);

    return R;
  }
  
  inline Matrix operator^(const Matrix&& A, const Value B)
  {
    return const_cast<Matrix&>(A) ^ B;
  }
  

  Matrix mul(const Matrix& A, const Matrix& B)
  {
    Matrix R(A.size().M, B.size().N, 0);
    for (size_t i = 0; i<B.size().N; i++)
      for (size_t j = 0; j<A.size().M; j++)
        for (size_t k = 0; k<A.size().N; k++)
          R[j][i] += A[j][k] * B[k][i];
    puts("mul end");
    return R;
  }
  
  inline Matrix mul(const Matrix& A, const Matrix&& B)
  {
    return mul(A, const_cast<Matrix&>(B));
  }
  
  inline Matrix mul(const Matrix&& A, const Matrix& B)
  {
    return mul(const_cast<Matrix&>(A), B);
  }
  
  inline Matrix mul(const Matrix&& A, const Matrix&& B)
  {
    return mul(const_cast<Matrix&>(A), const_cast<Matrix&>(B));
  }

  Matrix transpose(const Matrix& A)
  {
    Matrix R(A.size().N, A.size().M);
    for (size_t y = 0; y < A.size().M; ++y)
      for (size_t x = 0; x < A.size().N; ++x)
        R[x][y] = A[y][x];
    return R;
  }

  inline Matrix transpose(Matrix&& A)
  {
    return transpose(static_cast<Matrix&>(A));
  }

  std::unique_ptr<Matrix[]> MGS(const Matrix& A)
  {
    size_t M = A.size().M;
    size_t N = A.size().N;
    Matrix V = A;
    std::unique_ptr<Matrix[]> QR(new Matrix[2]{Matrix(M, N), Matrix(N, N, 0)});
    Matrix& Q = QR[0];
    Matrix& R = QR[1];

    size_t i, j, k;
    for (j = 0; j < N; ++j) {
      // q_j = v_j / ||v_j||_2
      Value sum_of_squares = 0;
      for (i = 0; i < M; ++i)
        sum_of_squares += V[i][j] * V[i][j];
      Value norm = std::sqrt(sum_of_squares);
      if (norm)
        for (i = 0; i < M; ++i)
          Q[i][j] = V[i][j] / norm;
      for (k = j; k < N; ++k) {
        // v_k = v_k - (qT_j*v_k)*q_j   <---------- to revise
        Value projection = 0;
        for (i = 0; i < M; ++i) projection += Q[i][j] * V[i][k];
        for (i = 0; i < M; ++i) V[i][k] -= projection * Q[i][j];
        // compute R
        if (k >= j) R[j][k] = projection;
      }
    }
    return QR;
  }

  inline std::unique_ptr<Matrix[]> MGS(const Matrix&& A)
  {
    return MGS(const_cast<Matrix&>(A));
  }

  std::unique_ptr<Matrix[]> FULL_MGS(const Matrix& A)
  {
    size_t i, j, k;
    size_t M = A.size().M;
    size_t N = A.size().N;
    Matrix V(M, M, 0);
    Matrix* QR = new Matrix[2]{Matrix(M, M, 0), Matrix(M, N, 0)};
    Value sum_of_squares, norm, projection;
    std::cout << "A:\n" << A << '\n';

    for (j = 0; j < M; ++j) {
      for (i = 0; i < M; ++i)
        if (j<N) V[i][j] = A[i][j];
    }

    for (j = 0; j < M; ++j) {
      // q_j = v_j / ||v_j||_2
      for (i = 0, sum_of_squares = 0; i < M; ++i)
        sum_of_squares += V[i][j] * V[i][j];
      norm = std::sqrt(sum_of_squares);
      if (norm) for (i = 0; i < M; ++i)
        QR[0][i][j] += V[i][j] / norm;
      for (k = j; k < M; ++k) {
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
  
  Matrix mldivide(Matrix& A, Matrix& B) {
    auto QR = MGS(A);

    Matrix Q = QR[0];
    Matrix R = QR[1];

    auto QtB = mul(transpose(Q), B);

    Matrix x(1, A.size().N, 0);

    for (size_t i = x.size().N - 1; i < x.size().N; --i) {
      x[0][i] = QtB[0][i];
      for (size_t j = x.size().N - 1; j > i; --j) {
        x[0][i] -= R[j][i] * x[0][j];
      }
      x[0][i] /= R[i][i];
    }

    return x;
  }

  std::unique_ptr<Matrix[]> linearize(const Matrix& xdata, const Matrix& ydata)
  {
    size_t N = xdata.size().numel;
    Matrix new_x(1, N, 0), new_y(1, N, 0);
    Value  step = (xdata[0][N-1] - xdata[0][0]) / (N - 1);
    std::cout << "step: " << step << '\n';
    Value  x1, x2, y1, y2;

    new_x[0][0]     = xdata[0][0];
    new_y[0][0]     = ydata[0][0];
    new_x[0][N - 1] = xdata[0][N - 1];
    new_y[0][N - 1] = ydata[0][N - 1];

    for (size_t index = 1; index < (N - 1); ++index) {
      new_x[0][index] = new_x[0][index-1] + step;

      x1 = xdata[0][index];
      y1 = ydata[0][index];
      x2 = xdata[0][index + 1];
      y2 = ydata[0][index + 1];

      new_y[0][index] = ((y1 - y2) * new_x[0][index] + x1 * y2 - x2 * y1) / (x1 - x2);
    }

    Matrix* resampled = new Matrix[2]{new_x, new_y};
    return std::unique_ptr<Matrix[]>(resampled);
  }
  
  inline std::unique_ptr<Matrix[]> linearize(const Matrix& xdata, const Matrix&& ydata)
  {
    return linearize(xdata, const_cast<Matrix&>(ydata));
  }
  
  inline std::unique_ptr<Matrix[]> linearize(const Matrix&& xdata, const Matrix& ydata)
  {
    return linearize(const_cast<Matrix&>(xdata), ydata);
  }
  
  inline std::unique_ptr<Matrix[]> linearize(const Matrix&& xdata, const Matrix&& ydata)
  {
    return linearize(const_cast<Matrix&>(xdata), const_cast<Matrix&>(ydata));
  }

  Matrix linspace(const Value x1, const Value x2, const size_t N)
  {
    Matrix vector(1, N);
    Value step = (x2 - x1) / (N - 1);
    vector[0][0]     = x1;
    vector[0][N - 1] = x2;

    for (size_t index = 1; index < (N - 1); ++index)
      vector[0][index] = vector[0][index-1] + step;

    return vector;
  }

  Matrix diff(const Matrix& A, size_t n)
  {
    if (n) {
      Matrix derivative(A.size().M, A.size().N - 1, 0);
      for (size_t y = 0; y < derivative.size().M; ++y)
        for (size_t x = 0; x < derivative.size().N; ++x)
          derivative[y][x] = A[y][x + 1] - A[y][x];

      return diff(derivative, n - 1);
    }
    return A;
  }

  inline Matrix diff(const Matrix&& A, size_t n)
  {
    return diff(const_cast<Matrix&>(A), n);
  }

  Matrix conv(const Matrix& A, const Matrix& B)
  {
    Matrix convoluted(1, A.size().N + B.size().N - 1);

    for (size_t x_A = 0; x_A < A.size().N; ++x_A)
      for (size_t x_B = 0; x_B < B.size().N; ++x_B)
        convoluted[0][x_A + x_B] += A[0][x_A] * B[0][x_B];

    return convoluted;
  }

  inline Matrix conv(const Matrix& A, const Matrix&& B)
  {
    return conv(A, const_cast<Matrix&>(B));
  }

  inline Matrix conv(const Matrix&& A, const Matrix& B)
  {
    return conv(const_cast<Matrix&>(A), B);
  }

  inline Matrix conv(const Matrix&& A, const Matrix&& B)
  {
    return conv(const_cast<Matrix&>(A), const_cast<Matrix&>(B));
  }

  Matrix blackman(const size_t L)
  {
    Matrix window(1, L);
    for (size_t index = 0; index < L; ++index)
      window[0][index] = 0.42 - 0.5*std::cos(2*M_PI*index/(L-1)) + 0.08*std::cos(4*M_PI*index/(L - 1));

    return window;
  }

  Matrix hamming(const size_t L)
  {
    Matrix window(1, L);
    for (size_t index = 0; index < L; ++index)
      window[0][index] = 0.54 - 0.46 * std::cos(2*M_PI*index/(L - 1));

    return window;
  }

  Matrix hann(const size_t L)
  {
    Matrix window(1, L);
    for (size_t index = 0; index < L; ++index)
      window[0][index] = 0.5*(1-cos(2*M_PI*index/(L-1)));

    return window;
  }

  Value newton(const Function function, const Value tol, const size_t max_iteration, const Value seed)
  {
    const Value half_tol = tol*0.5;

    Value root = seed;

    size_t iteration = 0;
    while ((tol < std::abs(function(root))) && (iteration < max_iteration)) {
      root -= tol * function(root) / (function(root + half_tol) - function(root - half_tol));
      ++iteration;
    }

    return root;
  }
}
// --Pinakas library: ostream overloads-------------------------------------------
namespace Pinakas::Backend
{
  std::ostream& operator<<(std::ostream& ostream, const Matrix& A)
  {
    if (A.size().numel) {
      // get longest number
      Value number = std::max(std::abs(min(A)), max(A));

      // find length of longest number in characters
      size_t length = std::ceil(std::log10(number)) + (min(A) < 0);

      // add matrix to ostream
      for (size_t y = 0; y < A.size().M; ++y) {
        for (size_t x = 0; x < A.size().N; ++x)
          ostream << std::setw(length) << A[y][x] << ' ';
        ostream << '\n';
      }
    }
    return ostream;
  }

  std::ostream& operator<<(std::ostream& ostream, const Matrix&& A)
  {
    return ostream << const_cast<Matrix&>(A);
  }

  inline std::ostream& operator<<(std::ostream& ostream, const Size size)
  {
    return ostream << size.N << 'x' << size.M;
  }
  
}