// --inclusion guard--------------------------------------------------------------
#include "../include/Pinakas.hpp"
#define M_PI 3.14159265358979323846
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  bool Size::operator==(const Size B) const noexcept
  {
    return (M == B.M) && (N == B.N) && (numel == B.numel);
  }

  bool Size::operator!=(const Size B) const noexcept 
  {
    return (M != B.M) || (N != B.N) || (numel != B.numel);
  }
  
  
  Matrix::~Matrix() noexcept
  {
    #ifdef LOGGING
    std::clog << "Matrix deleted !\n";
    #endif
  }

  
  Matrix::Matrix() noexcept
    : // member initialization list
    size_{0, 0, 0},
    memory_block_(nullptr),
    data_(nullptr)
  {
    #ifdef LOGGING
    std::clog << "Matrix created ! (empty)\n";
    #endif
  }

  
  
  Matrix::Matrix(const Matrix& matrix)
  {
    #ifdef LOGGING
    std::clog << "Matrix copied !\n";
    #endif
    // allocate memory
    allocate(matrix.size_.M, matrix.size_.N);
    // assign value to matrix
    for (size_t index = 0; index < size_.numel; ++index)
      data_[0][index] = matrix[0][index];
  }

  
  
  Matrix::Matrix(Matrix&& other) noexcept
    : // member initialization list
    size_(other.size_),
    memory_block_(other.memory_block_.release()),
    data_(other.data_)
  {
    #ifdef LOGGING
    std::clog << "Matrix moved !\n";
    #endif
  }

  
  
  Matrix::Matrix(const size_t M, const size_t N)
  {
    #ifdef LOGGING
    std::clog << "Matrix created !\n";
    #endif
    // allocate memory
    allocate(M, N);
  }

  
  Matrix::Matrix(const Size size)
    : Matrix(size.M, size.N)
  {}

  
  Matrix::Matrix(const size_t M, const size_t N, const double value)
  {
    #ifdef LOGGING
    std::clog << "Matrix created !\n";
    #endif
    // allocate memory
    allocate(M, N);
    // assign value to matrix
    for (size_t index = 0; index < size_.numel; ++index)
      data_[0][index] = value;
  }
  
  
  Matrix::Matrix(const Size size, double value)
    : Matrix(size.M, size.N, value)
  {}

  
  Matrix::Matrix(const size_t M, const size_t N, const double min, const double max)
  {
    #ifdef LOGGING
    std::clog << "Matrix created !\n";
    #endif
    // allocate memory
    allocate(M, N);
    // random number generator
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(min, max);
    // assign random value to matrix
    for (size_t index = 0; index < size_.numel; ++index)
      data_[0][index] = uniform_distribution(generator);
  }

  
  Matrix::Matrix(const Size size, double min, double max)
    : Matrix(size.M, size.N, min, max)
  {}


  
  Matrix::Matrix(const List<double> list)
  {
    #ifdef LOGGING
    std::clog << "Matrix created !\n";
    #endif
    // allocate memory
    allocate(1, list.size());
    // assign values into matrix
    size_t x = 0;
    for (double value : list)
      data_[0][x++] = value;
  }

  
  Matrix::Matrix(const List<const List<const double>> values)
  {
    #ifdef LOGGING
    std::clog << "Matrix created !\n";
    #endif
    // dimension validation
    size_t temp_N = 0;
    for (List<const double> vector : values) {
      if (temp_N && (temp_N != vector.size())) {
        std::cerr << "vertical dimensions mismatch (" << temp_N << " vs " << vector.size() << ")\n";
        size_ = {0, 0, 0};
        return;
      }
      else temp_N = vector.size();
    }
    // allocate memory
    allocate(values.size(), temp_N);
    // assign values into matrix
    size_t y = 0;
    for (List<const double> vector : values) {
      size_t x = 0;
      for (double value : vector) {
        data_[y][x] = value;
        ++x;
      }
      ++y;
    }
  }
  
  
  Matrix::Matrix(const List<const Matrix> list)
  {
    #ifdef LOGGING
    std::clog << "Matrix created !\n";
    #endif
    // dimension validation
    size_t temp_M = 0;
    size_t temp_N = 0;
    for (Matrix matrix : list) {
      if (temp_M && (temp_M != matrix.size_.M)) {
        std::cerr << "vertical dimensions mismatch (" << temp_M << " vs " << matrix.size_.M << ")\n";
        size_ = {0, 0, 0};
        return;
      }
      else temp_M = matrix.size_.M;
      temp_N += matrix.size_.N;
    }
    // allocate memory
    allocate(temp_M, temp_N);
    size_t index = 0;
    for (Matrix matrix : list) {
      for (size_t y = 0; y < matrix.size_.M; ++y)
        for (size_t x = 0; x < matrix.size_.N; ++x)
          data_[y][x + index] = matrix[y][x];
      index += matrix.size_.N;
    }
  }

  
  void Matrix::allocate(const size_t M, const size_t N)
  {
    // allocate memory
    memory_block_.reset(new char[sizeof(double*[M]) + sizeof(double[M][N])]);
    // get address of memory block
    char* address = memory_block_.get();
    // validate memory allocation
    if (!address) throw std::bad_alloc();
    // create rows into memory block
    data_ = (double**) address;
    // offset address
    address += sizeof(double*[M]);
    for (size_t y = 0; y < M; ++y) {
      // create columns into memory block
      data_[y] = (double*) address;
      // offset address
      address += sizeof(double[N]);
    }
    // save sizeValue
    size_ = {M, N, M*N};
  }

  
  double* Matrix::operator[](const size_t index) const noexcept
  {
    return data_[index];
  }

  
  double& Matrix::operator()(const size_t index) const
  {
    if (index >= size_.numel) {
      std::stringstream error_message;
      error_message << '(' << index << ") out of bound " << size_.numel - 1<< " (dimensions are " << size_ << ')';
      throw std::out_of_range(error_message.str());
    }
    return data_[0][index];
  }

  
  double& Matrix::operator()(const size_t y, const size_t x) const 
  {
    if (y >= size_.M) {
      std::stringstream error_message;
      error_message << '(' << y << ",_) out of bound " << size_.M - 1<< " (dimensions are " << size_ << ')';
      throw std::out_of_range(error_message.str());
    }
    if (x >= size_.N) {
      std::stringstream error_message;
      error_message << "(_," << x << ") out of bound " << size_.N << " (dimensions are " << size_ << ')';
      throw std::out_of_range(error_message.str());
    }
    return data_[y][x];
  }
  
  Size Matrix::size(void) const noexcept
  {
    return size_;
  }
// -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string& op)
  {
    if (size_A != size_B) {
      std::stringstream error_message;
      error_message << "operator " << op << ": nonconformant arguments (";
      error_message << "A is " << size_A.M << 'x' << size_A.N;
      error_message << ", B is " << size_B.M << 'x' << size_B.N << ")\n";
      throw std::invalid_argument(error_message.str());
    }
  }
// -------------------------------------------------------------------------------

  
  Matrix& Matrix::operator=(const Matrix& B)
  {
    std::cout << "assigned\n";
    if (this != &B) {
      if (size_ != B.size_) allocate(B.size_.M, B.size_.N);
      if (memory_block_.get()) {
        for (size_t index = 0; index < size_.numel; ++index)
          data_[0][index] = B[0][index];
      }
    }
    return *this;
  }

  
  Matrix& Matrix::operator=(Matrix&& B) noexcept
  {
    std::cout << "move assigned\n";
    size_ = B.size_;
    memory_block_.reset(B.memory_block_.release());
    data_ = B.data_;
    return *this;
  }
// -------------------------------------------------------------------------------
  
  Matrix& Matrix::operator=(const double B) noexcept
  {
    for (size_t index = 0; index < size_.numel; ++index)
      data_[0][index] = B;
    return *this;
  }
// -------------------------------------------------------------------------------
  
  Matrix& operator+=(Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "+=");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += B[0][index];
    return A;
  }

  
  Matrix operator+(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "+");
    Matrix R(A.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = A[0][index] + B[0][index];
    return R;
  }

  
  Matrix&& operator+(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] += A[0][index];
    return std::move(B);
  }

  
  Matrix&& operator+(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += B[0][index];
    return std::move(A);
  }

  
  Matrix&& operator+(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += B[0][index];
    return std::move(A);
  }

  
  Matrix& operator+=(Matrix& A, const double B) noexcept
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += B;
    return A;
  }

  
  Matrix operator+(const Matrix& A, const double B) noexcept
  {
    Matrix R(A.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = A[0][index] + B;
    return R;
  }

  
  Matrix&& operator+(Matrix&& A, const double B) noexcept
  {
    return std::move(A += B);
  }

  
  Matrix operator+(const double A, const Matrix& B) noexcept
  {
    return B + A;
  }

  
  Matrix&& operator+(const double A, Matrix&& B) noexcept
  {
    return std::move(B += A);
  }

  
  Matrix operator+(const Matrix& A)
  {
    return A;
  }

  
  Matrix&& operator+(Matrix&& A) noexcept
  {
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  
  Matrix& operator*=(Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "*=");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] *= B[0][index];
    return A;
  }

  
  Matrix operator*(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "*");
    Matrix R(A.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = A[0][index] * B[0][index];
    return R;
  }

  
  Matrix&& operator*(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] *= A[0][index];
    return std::move(B);
  }

  
  Matrix&& operator*(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] *= B[0][index];
    return std::move(A);
  }

  
  Matrix&& operator*(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] *= B[0][index];
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  
  Matrix& operator*=(Matrix& A, const double B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] *= B;
    return A;
  }

  
  Matrix&& operator*(Matrix&& A, const double B) noexcept
  {
    return std::move(A *= B);
  }

  
  Matrix operator*(const double A, const Matrix& B) noexcept
  {
    return B * A;
  }

  
  Matrix&& operator*(const double A, Matrix&& B) noexcept
  {
    return std::move(B *= A);
  }
// -------------------------------------------------------------------------------
  
  Matrix& operator-=(Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "-=");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] -= B[0][index];
    return A;
  }
  
  
  Matrix operator-(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "-");
    Matrix R(A.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = A[0][index] - B[0][index];
    return R;
  }

  
  Matrix&& operator-(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] = A[0][index] - B[0][index];
    return std::move(B);
  }

  
  Matrix&& operator-(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] -= B[0][index];
    return std::move(A);
  }

  
  Matrix&& operator-(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < B.size().numel; ++index)
      A[0][index] -= B[0][index];
    return std::move(A);
  }

  
  Matrix& operator-=(Matrix& A, const double B) noexcept
  {
    return A += (-B);
  }

  
  Matrix operator-(const Matrix& A, const double B) noexcept
  {
    return A + (-B);
  }

  
  Matrix&& operator-(Matrix&& A, const double B) noexcept
  {
    return std::move(A += (-B));
  }

  
  Matrix operator-(const double A, const Matrix& B) noexcept
  {
    Matrix R(B.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = A - B[0][index];
    return R;
  }

  
  Matrix&& operator-(const double A, Matrix&& B) noexcept
  {
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] = A - B[0][index];
    return std::move(B);
  }

  
  Matrix operator-(const Matrix& A)
  {
    Matrix R(A.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = -A[0][index];
    return R;
  }

  
  Matrix&& operator-(Matrix&& A) noexcept
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = -A[0][index];
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  
  Matrix& operator/=(Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "/=");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] /= B[0][index];
    return A;
  }
  
  
  Matrix operator/(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "/");
    Matrix R(A.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = A[0][index] / B[0][index];
    return R;
  }

  
  Matrix&& operator/(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] = A[0][index] / B[0][index];
    return std::move(B);
  }

  
  Matrix&& operator/(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] /= B[0][index];
    return std::move(A);
  }

  
  Matrix&& operator/(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] /= B[0][index];
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  
  Matrix& operator/=(Matrix& A, const double B) noexcept
  {
    return A *= (1 / B);
  }

  
  Matrix operator/(const Matrix& A, const double B) noexcept
  {
    return A * (1 / B);
  }

  
  Matrix&& operator/(Matrix&& A, const double B) noexcept
  {
    return std::move(A *= (1 / B));
  }

  
  Matrix operator/(const double A, const Matrix& B) noexcept
  {
    Matrix R(B.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = A / B[0][index];
    return R;
  }

  
  Matrix&& operator/(const double A, Matrix&& B) noexcept
  {
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] = A / B[0][index];
    return std::move(B);
  }
// -------------------------------------------------------------------------------
  
  Matrix& operator^=(Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "^=");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return A;
  }
  
  
  Matrix operator^(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "^");
    Matrix R(A.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = std::pow(A[0][index], B[0][index]);
    return R;
  }

  
  Matrix&& operator^(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(B);
  }

  
  Matrix&& operator^(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(A);
  }

  
  Matrix&& operator^(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  
  Matrix& operator^=(Matrix& A, const double B) noexcept
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::pow(A[0][index], B);
    return A;
  }

  
  Matrix operator^(const Matrix& A, const double B) noexcept
  {
    Matrix R(A.size());
    for (size_t index = 0; index < A.size().numel; ++index)
      R[0][index] = std::pow(A[0][index], B);
    return R;
  }

  
  Matrix&& operator^(Matrix&& A, const double B) noexcept
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::pow(A[0][index], B);
    return std::move(A);
  }

  
  Matrix operator^(const double A, const Matrix& B) noexcept
  {
    Matrix R(B.size());
    for (size_t index = 0; index < B.size().numel; ++index)
      R[0][index] = std::pow(A, B[0][index]);
    return R;
  }

  
  Matrix&& operator^(const double A, Matrix&& B) noexcept
  {
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] = std::pow(A, B[0][index]);
    return std::move(B);
  }
// -------------------------------------------------------------------------------
  
  Matrix floor(const Matrix& A)
  {
    Matrix R(A.size(), 0);
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = std::floor(A[0][index]);
    return R;
  }

  
  Matrix&& floor(Matrix&& A) noexcept
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::floor(A[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  
  Matrix round(const Matrix& A)
  {
    Matrix R(A.size(), 0);
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = std::round(A[0][index]);
    return R;
  }
  
  
  Matrix&& round(Matrix&& A) noexcept
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::round(A[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  
  Matrix ceil(Matrix& A)
  {
    Matrix R(A.size(), 0);
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = std::ceil(A[0][index]);
    return R;
  }
  
  
  Matrix&& ceil(Matrix&& A) noexcept
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::round(A[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  
  Matrix mul(const Matrix& A, const Matrix& B)
  {
    if (A.size().N != B.size().M) {
      std::stringstream error_message;
      error_message << "operation mul : nonconformant arguments (";
      error_message << "A is " << A.size().M << 'x' << A.size().N;
      error_message << ", B is " << B.size().M << 'x' << B.size().N << ")\n";
      throw std::invalid_argument(error_message.str());
    }
    Matrix R(A.size().M, B.size().N, 0);
    for (size_t i = 0; i<B.size().N; i++)
      for (size_t j = 0; j<A.size().M; j++)
        for (size_t k = 0; k<A.size().N; k++)
          R[j][i] += A[j][k] * B[k][i];
    return R;
  }
// -------------------------------------------------------------------------------
  
  Matrix transpose(const Matrix& A)
  {
    Matrix R(A.size().N, A.size().M);
    for (size_t y = 0; y < A.size().M; ++y)
      for (size_t x = 0; x < A.size().N; ++x)
        R[x][y] = A[y][x];
    return R;
  }

  Matrix transpose(Matrix&& A)
  {
    return transpose(A);
  }
// -------------------------------------------------------------------------------
  
  double min(const Matrix& matrix) noexcept
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t index = 0; index < matrix.size().numel; ++index)
      if (matrix[0][index] < minimum) minimum = matrix[0][index];
    return minimum;
  }
// -------------------------------------------------------------------------------
  
  double max(const Matrix& matrix) noexcept
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t index = 0; index < matrix.size().numel; ++index)
      if (matrix[0][index] > maximum) maximum = matrix[0][index];
    return maximum;
  }
// -------------------------------------------------------------------------------
  
  double sum(const Matrix& matrix) noexcept
  {
    double summation = 0;
    for (size_t index = 0; index < matrix.size().numel; ++index)
      summation += matrix[0][index];
    return summation;
  }
// -------------------------------------------------------------------------------
  
  double prod(const Matrix& matrix) noexcept
  {
    double product = 0;
    for (size_t index = 0; index < matrix.size().numel; ++index)
      product *= matrix[0][index];
    return product;
  }
// -------------------------------------------------------------------------------
  std::unique_ptr<Matrix[]> MGS(const Matrix& A)
  {
    size_t M = A.size().M;
    size_t N = A.size().N;
    Matrix V = A;
    std::unique_ptr<Matrix[]> QR(new Matrix[2]{{Matrix(M, N)}, Matrix(N, N, 0)});
    Matrix& Q = QR[0];
    Matrix& R = QR[1];

    size_t i, j, k;
    for (j = 0; j < N; ++j) {
      // q_j = v_j / ||v_j||_2
      double sum_of_squares = 0;
      for (i = 0; i < M; ++i)
        sum_of_squares += V[i][j] * V[i][j];
      double norm = std::sqrt(sum_of_squares);
      if (norm)
        for (i = 0; i < M; ++i)
          Q[i][j] = V[i][j] / norm;
      for (k = j; k < N; ++k) {
        // v_k = v_k - (qT_j*v_k)*q_j   <---------- to revise
        double projection = 0;
        for (i = 0; i < M; ++i) projection += Q[i][j] * V[i][k];
        for (i = 0; i < M; ++i) V[i][k] -= projection * Q[i][j];
        // compute R
        if (k >= j) R[j][k] = projection;
      }
    }
    return QR;
  }

  Matrix div(Matrix& A, Matrix& B) {
    auto M = B.size().M;
    auto N = B.size().N;

    auto V = B;
    auto Q = Matrix(M, N, 0);
    auto R = Matrix(N, N, 0);

    for (size_t x = 0; x < N; ++x) {
      double sum_of_squares = 0;
      for (size_t y = 0; y < M; ++y)
        sum_of_squares += V(y, x) * V(y, x);
      double norm = sqrt(sum_of_squares);
      if (norm)
        for (size_t y = 0; y < M; ++y)
          Q(y, x) = V(y, x) / norm;
      for (size_t k = x; k < N; ++k) {
        double projection = 0;
        for (size_t y = 0; y < M; ++y)
          projection += Q(y, x) * V(y, k);
        for (size_t y = 0; y < M; ++y)
          V(y, k) -= projection * Q(y, x);
        if (k >= x)
          R(x, k) = projection;
      }
    }

    Matrix Qt(N, M);
    for (size_t y = 0; y < M; ++y)
      for (size_t x = 0; x < N; ++x)
        Qt[x][y] = Q[y][x];

    Matrix QtA(N, 1, 0);
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < M; ++k)
        QtA[j][0] += Qt[j][k] * A[k][0];

    //*/
    Matrix x(N, 1);
    //*
    for (size_t i = N-1; i < N; --i) {
      double temp = QtA(i, 0);
      for (size_t j = N-1; j > i; --j)
        temp -= R(i, j) * x(j, 0);
      x(i, 0) = temp / R(i, i);
    }
    //*/

    return x;
  }

  
  std::unique_ptr<Matrix[]> linearize(const Matrix& xdata, const Matrix& ydata)
  {
    size_t N = xdata.size().numel;
    Matrix new_x(1, N, 0), new_y(1, N, 0);
    double  step = (xdata[0][N-1] - xdata[0][0]) / (N - 1);
    std::cout << "step: " << step << '\n';
    double  x1, x2, y1, y2;

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
  
  
  std::unique_ptr<Matrix[]> linearize(const Matrix& xdata, Matrix&& ydata)
  {
    return linearize(xdata, (ydata));
  }
  
  
  std::unique_ptr<Matrix[]> linearize(Matrix&& xdata, const Matrix& ydata)
  {
    return linearize((xdata), ydata);
  }
  
  
  std::unique_ptr<Matrix[]> linearize(Matrix&& xdata, Matrix&& ydata)
  {
    size_t N = xdata.size().numel;
    Matrix new_x(1, N, 0), new_y(1, N, 0);
    double  step = (xdata[0][N-1] - xdata[0][0]) / (N - 1);
    std::cout << "step: " << step << '\n';
    double  x1, x2, y1, y2;

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

  Matrix linspace(const double x1, const double x2, const size_t N)
  {
    Matrix vector(1, N);
    double step = (x2 - x1) / (N - 1);
    vector[0][0]     = x1;
    vector[0][N - 1] = x2;
    for (size_t index = 1; index < (N - 1); ++index)
      vector[0][index] = vector[0][index-1] + step;
    return vector;
  }

  Matrix iota(const size_t N)
  {
    Matrix vector(1, N);
    for (size_t index = 0; index < vector.size().numel ; ++index)
      vector[0][index] = index;
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

  
  Matrix conv(const Matrix& A, const Matrix& B)
  {
    Matrix convoluted(1, A.size().N + B.size().N - 1);
    for (size_t x_A = 0; x_A < A.size().N; ++x_A)
      for (size_t x_B = 0; x_B < B.size().N; ++x_B)
        convoluted[0][x_A + x_B] += A[0][x_A] * B[0][x_B];
    return convoluted;
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

  
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed)
  {
    const double half_tol = tol*0.5;

    double root = seed;

    size_t iteration = 0;
    while ((tol < std::abs(function(root))) && (iteration++ < max_iteration))
      root -= tol * function(root) / (function(root + half_tol) - function(root - half_tol));

    return root;
  }

  double sinc(double x, double freq = 1) {
    return x != 0 ? sin(M_PI*x*freq)/(M_PI*x*freq) : 1;
  }

  Matrix sinc(Matrix& A, double freq = 1) {
    Matrix R(A.size());
    for (size_t index = 0; index < R.size().numel; ++index)
      R[0][index] = sinc(A[0][index], freq);
    return R;
  }

  Matrix&& sinc(Matrix&& A, double freq = 1) {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = sinc(A[0][index], freq);
    return std::move(A);
  }

  //
  Matrix resample(Matrix data, int L) {
	int N = data.size().numel*L; // approximate new data lenght
	int o = 3 * L;     // offset for filter impulse
	int l = 2 * o + 1; // filter impulse lenght

	Matrix u(1, N, 0); // upsampled data
	for (size_t i = 0; i < data.size().numel; ++i) {
		u[0][i*L] = data[0][i];
	}

  auto temp = iota(l)-o;
  std::cout << temp;
  auto h = sinc(temp, 1/L); // windowed filter impulse
  std::cout << h;
	auto w = blackman(l); // upsampled data
  auto hw = h * w;
	Matrix r(1, N-L+1); // cropped convolution of upsampled data with windowed filter impulse
	for (int i = 0; i<N-L+1; i++)
		for (int n = 0; n<l; n++)
			r[0][i] += u[0][(i+n-o) < N ? std::abs(i+n-o) : (2*N-L - (i+n-o))] * hw[0][n];

	return r;
}





}
// --Pinakas library: ostream overloads-------------------------------------------
namespace Pinakas::Backend
{
  
  std::ostream& operator<<(std::ostream& ostream, const Matrix& A)
  {
    if (A.size().numel) {
      // get longest number
      //double number = std::max(std::abs(min(A)), max(A));

      // find length of longest number in characters
      //size_t length = std::ceil(std::log10(number)) + (min(A) < 0);
      // add matrix to ostream
      for (size_t y = 0; y < A.size().M; ++y) {
        for (size_t x = 0; x < A.size().N; ++x)
          ostream << A[y][x] << ' ';
          //ostream << std::setw(length) << A[y][x] << ' ';
        ostream << '\n';
      }
    }
    return ostream;
  }
  
  std::ostream& operator<<(std::ostream& ostream, Matrix&& A)
  {
    return ostream << A;
  }

  std::ostream& operator<<(std::ostream& ostream, const Size size)
  {
    return ostream << size.N << 'x' << size.M;
  }
  
}
// -
int main()
{
  using namespace Pinakas;
  Matrix x = transpose(iota(4) + 1);
  Matrix y = 2*(x^2) - x + 3;
  Matrix w = {{x^2}, {x}, {x^0}};

  std::cout << div(y, w);
  //std::cout << x.size();
  //std::cout << x;
  //resample(x, 2);
  //std::cout << "x:\n" << resample(x, 2);
}