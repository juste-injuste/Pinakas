// --inclusion guard--------------------------------------------------------------
#include "../include/Pinakas.hpp"
#define M_PI 3.14159265358979323846
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas { namespace Backend
{
  bool Size::operator==(const Size B) const
  {
    return (M == B.M) && (N == B.N) && (numel == B.numel);
  }

  bool Size::operator!=(const Size B) const
  {
    return (M != B.M) || (N != B.N) || (numel != B.numel);
  }

  Matrix::~Matrix()
  {
    #ifdef LOGGING
    std::clog << "Matrix deleted !\n";
    #endif
  }

  Matrix::Matrix()
    : // member initialization list
    size_{0, 0, 0},
    data_(nullptr)
  {
#ifdef LOGGING
    std::clog << "Matrix created ! (empty)\n";
#endif
  }

  Matrix::Matrix(const Matrix &other)
  {
#ifdef LOGGING
    std::clog << "Matrix copied !\n";
#endif

    if (this != &other) {
      // allocate memory
      allocate(this, other.size_.M, other.size_.N);

      // store value
      for (size_t index = 0; index < size_.numel; ++index)
        data_[index] = other[0][index];
    }
  }

  Matrix::Matrix(Matrix &&other)
    : // member initialization list
    size_(other.size_),
    data_(other.data_.release())
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
    allocate(this, M, N);
  }

  Matrix::Matrix(const Size size)
    : // member initialization list
    Matrix(size.M, size.N)
  {}

  Matrix::Matrix(const size_t M, const size_t N, const double value)
  {
#ifdef LOGGING
    std::clog << "Matrix created !\n";
#endif

    // allocate memory
    allocate(this, M, N);

    // store values
    for (size_t index = 0; index < size_.numel; ++index)
      data_[index] = value;
  }

  Matrix::Matrix(const Size size, double value)
    : Matrix(size.M, size.N, value)
  {}

  Matrix::Matrix(const size_t M, const size_t N, const Random range)
  {
#ifdef LOGGING
    std::clog << "Matrix created !\n";
#endif

    // allocate memory
    allocate(this, M, N);

    // random number generator
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);

    // assign random value to matrix
    for (size_t index = 0; index < size_.numel; ++index)
      data_[index] = uniform_distribution(generator);
  }

  Matrix::Matrix(const Size size, const Random range)
    : // member initialization list
    Matrix(size.M, size.N, range)
  {}

  Matrix::Matrix(const List<double> list)
  {
#ifdef LOGGING
    std::clog << "Matrix created !\n";
#endif

    // allocate memory
    allocate(this, 1, list.size());

    // store values
    size_t x = 0;
    for (double value : list)
      data_[x++] = value;
  }

  Matrix::Matrix(const List<const List<const double>> values)
  {
#ifdef LOGGING
    std::clog << "Matrix created !\n";
#endif

    // dimension validation
    size_t temp_N = 0;
    for (const List<const double> &vector : values) {
      if (temp_N && (temp_N != vector.size())) {
        std::cerr << "error: vertical dimensions mismatch (" << temp_N << " vs " << vector.size() << ")\n";
        size_ = {0, 0, 0};
        return;
      }
      else
        temp_N = vector.size();
    }

    // allocate memory
    allocate(this, values.size(), temp_N);

    // store values
    size_t y = 0;
    for (const List<const double> &vector : values) {
      size_t x = 0;
      for (double value : vector) {
        data_[x + y * size_.N] = value;
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
    for (const Matrix &matrix : list) {
      if (temp_M && (temp_M != matrix.size_.M)) {
        std::cerr << "error: horizontal dimensions mismatch (" << temp_M << " vs " << matrix.size_.M << ")\n";
        size_ = {0, 0, 0};
        return;
      }
      else
        temp_M = matrix.size_.M;
      temp_N += matrix.size_.N;
    }

    // allocate memory
    allocate(this, temp_M, temp_N);

    // store values
    size_t index = 0;
    for (const Matrix &matrix : list) {
      for (size_t y = 0; y < matrix.size_.M; ++y)
        for (size_t x = 0; x < matrix.size_.N; ++x)
          data_[x + index + y * size_.N] = matrix[y][x];
      index += matrix.size_.N;
    }
  }

  void allocate(Matrix *matrix, const size_t M, const size_t N)
  {
    // validate sizes
    if ((M == 0) || (N == 0)) {
      std::stringstream error_message;
      error_message << "error: allocate: dimensions are " << M << 'x' << N;
      throw std::invalid_argument(error_message.str());
    }

    // allocate memory
    matrix->data_.reset((double *)new char[sizeof(double[M][N])]);

    // validate memory allocation
    if (!matrix->data_.get())
      throw std::bad_alloc();

    // save size information
    matrix->size_ = {M, N, M * N};
  }

  double *Matrix::operator[](const size_t index) const
  {
    return data_.get() + index * size_.N;
  }

  double &Matrix::operator()(const size_t index) const
  {
    if (index >= size_.numel) {
      std::stringstream error_message;
      error_message << '(' << index << ") out of bound " << size_.numel - 1 << " (dimensions are " << size_ << ')';
      throw std::out_of_range(error_message.str());
    }
    return data_[index];
  }

  double &Matrix::operator()(Keyword::End) const
  {
    return data_[size_.numel - 1];
  }

  double &Matrix::operator()(const size_t y, const size_t x) const
  {
    if (y >= size_.M) {
      std::stringstream error_message;
      error_message << '(' << y << ",_) out of bound " << size_.M - 1 << " (dimensions are " << size_ << ')';
      throw std::out_of_range(error_message.str());
    }
    if (x >= size_.N) {
      std::stringstream error_message;
      error_message << "(_," << x << ") out of bound " << size_.N << " (dimensions are " << size_ << ')';
      throw std::out_of_range(error_message.str());
    }
    return data_[x + y * size_.N];
  }

  Slice Matrix::operator()(Keyword::Entire, const size_t n) &
  {
    return Slice(*this, n, Keyword::column);
  }

  Slice Matrix::operator()(const size_t m, Keyword::Entire) &
  {
    return Slice(*this, m, Keyword::row);
  }

  Size Matrix::size(void) const &
  {
    return size_;
  }

  size_t Matrix::numel(void) const &
  {
    return size_.numel;
  }

  size_t Matrix::M(void) const &
  {
    return size_.M;
  }

  size_t Matrix::N(void) const &
  {
    return size_.N;
  }
  // -------------------------------------------------------------------------------
  template <typename T>
  Iterator<T>::Iterator(T &matrix, const size_t index)
    : // member initialization list
    matrix(matrix),
    index(index)
  {}

  template <typename T>
  bool Iterator<T>::operator==(const Iterator<T> &other) const
  {
    return index == other.index;
  }

  template <typename T>
  bool Iterator<T>::operator!=(const Iterator<T> &other) const
  {
    return this->index != other.index;
  }

  template <typename T>
  Iterator<T> &Iterator<T>::operator++(void)
  {
    ++index;
    return *this;
  }

  template <typename T>
  double &Iterator<T>::operator*(void) const
  {
    return matrix[0][index];
  }
  // -------------------------------------------------------------------------------
  template <typename T>
  ConstIterator<T>::ConstIterator(const T &matrix, const size_t index)
    : // member initialization list
    matrix(matrix), index(index)
  {}

  template <typename T>
  bool ConstIterator<T>::operator==(const ConstIterator<T> &other) const
  {
    return index == other.index;
  }

  template <typename T>
  bool ConstIterator<T>::operator!=(const ConstIterator<T> &other) const
  {
    return this->index != other.index;
  }

  template <typename T>
  ConstIterator<T> &ConstIterator<T>::operator++()
  {
    ++index;
    return *this;
  }

  template <typename T>
  const double &ConstIterator<T>::operator*(void) const
  {
    return matrix[0][index];
  }
  // -------------------------------------------------------------------------------
  Iterator<Matrix> Matrix::begin(void)
  {
    return Iterator<Matrix>(*this, 0);
  }

  Iterator<Matrix> Matrix::end(void)
  {
    return Iterator<Matrix>(*this, size_.numel);
  }

  ConstIterator<Matrix> Matrix::begin(void) const
  {
    return ConstIterator<Matrix>(*this, 0);
  }

  ConstIterator<Matrix> Matrix::end(void) const
  {
    return ConstIterator<Matrix>(*this, size_.numel);
  }
  // -------------------------------------------------------------------------------
  Matrix &Matrix::operator=(const Matrix &other) &
  {
#ifdef LOGGING
    std::clog << "assigned\n";
#endif

    // validate both matrices are not the same
    if (this != &other) {
      // allocate memory if necessary
      if ((size_ != other.size_) || !data_.get())
        allocate(this, other.size_.M, other.size_.N);

      // store values
      for (size_t index = 0; index < size_.numel; ++index)
        data_[index] = other[0][index];
    }
    return *this;
  }

  Matrix &Matrix::operator=(Matrix &&other) &
  {
#ifdef LOGGING
    std::clog << "move assigned\n";
#endif

    // take over ressources from other matrix
    size_ = other.size_;
    data_.reset(other.data_.release());
    return *this;
  }
// -------------------------------------------------------------------------------

  Matrix &Matrix::operator=(const double value) &
  {
    // store values
    for (size_t index = 0; index < size_.numel; ++index)
      data_[index] = value;
    return *this;
  }
// -------------------------------------------------------------------------------
  Slice::Slice(Matrix &matrix, const size_t n, Keyword::Column)
    : // member initialization list
    size_{matrix.size().M, 1, matrix.size().M},
    fixed_(n),
    col_row_(false),
    matrix_(matrix)
  {}

  Slice::Slice(Matrix &matrix, const size_t m, Keyword::Row)
    : // member initialization list
    size_{1, matrix.size().N, matrix.size().N},
    fixed_(m),
    col_row_(true),
    matrix_(matrix)
  {}

  double &Slice::operator[](const size_t index) const &
  {
    return col_row_ ? matrix_[fixed_][index] : matrix_[index][fixed_];
  }

  double &Slice::operator()(const size_t index) const &
  {
    // validate index
    if (index >= size().numel) {
      std::stringstream error_message;
      error_message << '(' << index << ") out of bound " << size().numel - 1 << " (dimensions are ";
      error_message << (col_row_ ? 1 : size().numel) << 'x' << (col_row_ ? size().numel : 1) << " )";
      throw std::out_of_range(error_message.str());
    }
    return col_row_ ? matrix_[fixed_][index] : matrix_[index][fixed_];
  }

  Size Slice::size(void) const &
  {
    return size_;
  }

  size_t Slice::numel(void) const &
  {
    return size_.numel;
  }
  // -------------------------------------------------------------------------------
  Random::Random(double min, double max)
    : // member initialization list
    min_(min),
    max_(max)
  {}
  // -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string &op)
  {
    if (size_A != size_B) {
      std::stringstream error_message;
      error_message << "error: operator " << op << ": nonconformant arguments (";
      error_message << "A is " << size_A.M << 'x' << size_A.N;
      error_message << ", B is " << size_B.M << 'x' << size_B.N << ")\n";
      throw std::invalid_argument(error_message.str());
    }
  }
  // -------------------------------------------------------------------------------
  Matrix &operator+=(Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "+=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] += B[0][index];
    return A;
  }

  Matrix operator+(const Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "+");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] + B[0][index];
    return result;
  }

  Matrix operator+(const Matrix &A, const Random range)
  {
    Matrix result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] += A[0][index] + uniform_distribution(generator);
    return result;
  }

  Matrix &&operator+(Matrix &&A, const Random range)
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] += uniform_distribution(generator);
    return std::move(A);
  }

  Matrix operator+(const Random range, const Matrix &A)
  {
    Matrix result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] += A[0][index] + uniform_distribution(generator);
    return result;
  }

  Matrix &&operator+(const Random range, Matrix &&A)
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] += uniform_distribution(generator);
    return std::move(A);
  }

  Matrix &&operator+(const Matrix &A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] += A[0][index];
    return std::move(B);
  }

  Matrix &&operator+(Matrix &&A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] += B[0][index];
    return std::move(A);
  }

  Matrix &&operator+(Matrix &&A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] += B[0][index];
    return std::move(A);
  }

  Matrix &operator+=(Matrix &A, const double B)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] += B;
    return A;
  }

  Matrix operator+(const Matrix &A, const double B)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] + B;
    return result;
  }

  Matrix &&operator+(Matrix &&A, const double B)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] += B;
    return std::move(A += B);
  }

  Matrix operator+(const double A, const Matrix &B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A + B[0][index];
    return result;
  }

  Matrix &&operator+(const double A, Matrix &&B)
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] += A;
    return std::move(B);
  }

  Matrix operator+(const Matrix &A)
  {
    return A;
  }

  Matrix &&operator+(Matrix &&A)
  {
    return std::move(A);
  }
  // -------------------------------------------------------------------------------
  Matrix &operator*=(Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "*=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B[0][index];
    return A;
  }

  Matrix operator*(const Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "*");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] * B[0][index];
    return result;
  }

  Matrix &&operator*(const Matrix &A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] *= A[0][index];
    return std::move(B);
  }

  Matrix &&operator*(Matrix &&A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B[0][index];
    return std::move(A);
  }

  Matrix &&operator*(Matrix &&A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B[0][index];
    return std::move(A);
  }
  // -------------------------------------------------------------------------------
  Matrix &operator*=(Matrix &A, const double B)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B;
    return A;
  }

  Matrix operator*(const Matrix &A, const double B)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] * B;
    return result;
  }

  Matrix &&operator*(Matrix &&A, const double B)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B;
    return std::move(A *= B);
  }

  Matrix operator*(const double A, const Matrix &B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A * B[0][index];
    return result;
  }

  Matrix &&operator*(const double A, Matrix &&B)
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] *= A;
    return std::move(B);
  }
  // -------------------------------------------------------------------------------
  Matrix &operator-=(Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "-=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= B[0][index];
    return A;
  }

  Matrix operator-(const Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "-");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] - B[0][index];
    return result;
  }

  Matrix &&operator-(const Matrix &A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = A[0][index] - B[0][index];
    return std::move(B);
  }

  Matrix &&operator-(Matrix &&A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= B[0][index];
    return std::move(A);
  }

  Matrix &&operator-(Matrix &&A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < B.numel(); ++index)
      A[0][index] -= B[0][index];
    return std::move(A);
  }

  Matrix &operator-=(Matrix &A, const double B)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= B;
    return A;
  }

  Matrix operator-(const Matrix &A, const double B)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] - B;
    return result;
  }

  Matrix &&operator-(Matrix &&A, const double B)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= B;
    return std::move(A);
  }

  Matrix operator-(const double A, const Matrix &B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A - B[0][index];
    return result;
  }

  Matrix &&operator-(const double A, Matrix &&B)
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = A - B[0][index];
    return std::move(B);
  }

  Matrix operator-(const Matrix &A)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = -A[0][index];
    return result;
  }

  Matrix &&operator-(Matrix &&A)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = -A[0][index];
    return std::move(A);
  }
  // -------------------------------------------------------------------------------
  Matrix &operator/=(Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "/=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] /= B[0][index];
    return A;
  }

  Matrix operator/(const Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "/");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] / B[0][index];
    return result;
  }

  Matrix &&operator/(const Matrix &A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = A[0][index] / B[0][index];
    return std::move(B);
  }

  Matrix &&operator/(Matrix &&A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] /= B[0][index];
    return std::move(A);
  }

  Matrix &&operator/(Matrix &&A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] /= B[0][index];
    return std::move(A);
  }
  // -------------------------------------------------------------------------------
  Matrix &operator/=(Matrix &A, const double B)
  {
    const double iB = 1 / B;
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= iB;
    return A;
  }

  Matrix operator/(const Matrix &A, const double B)
  {
    const double iB = 1 / B;
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] * iB;
    return result;
  }

  Matrix &&operator/(Matrix &&A, const double B)
  {
    const double iB = 1 / B;
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= iB;
    return std::move(A);
  }

  Matrix operator/(const double A, const Matrix &B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A / B[0][index];
    return result;
  }

  Matrix &&operator/(const double A, Matrix &&B)
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = A / B[0][index];
    return std::move(B);
  }
  // -------------------------------------------------------------------------------
  Matrix &operator^=(Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "^=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return A;
  }

  Matrix operator^(const Matrix &A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "^");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::pow(A[0][index], B[0][index]);
    return result;
  }

  Matrix &&operator^(const Matrix &A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(B);
  }

  Matrix &&operator^(Matrix &&A, const Matrix &B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(A);
  }

  Matrix &&operator^(Matrix &&A, Matrix &&B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(A);
  }
  // -------------------------------------------------------------------------------
  Matrix &operator^=(Matrix &A, const double B)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B);
    return A;
  }

  Matrix operator^(const Matrix &A, const double B)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::pow(A[0][index], B);
    return result;
  }

  Matrix &&operator^(Matrix &&A, const double B)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B);
    return std::move(A);
  }

  Matrix operator^(const double A, const Matrix &B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::pow(A, B[0][index]);
    return result;
  }

  Matrix &&operator^(const double A, Matrix &&B)
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = std::pow(A, B[0][index]);
    return std::move(B);
  }
  // -------------------------------------------------------------------------------
  Matrix floor(const Matrix &A)
  {
    Matrix result(A.size(), 0);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::floor(A[0][index]);
    return result;
  }

  Matrix &&floor(Matrix &&A)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::floor(A[0][index]);
    return std::move(A);
  }
  // -------------------------------------------------------------------------------
  Matrix round(const Matrix &A)
  {
    Matrix result(A.size(), 0);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::round(A[0][index]);
    return result;
  }

  Matrix &&round(Matrix &&A)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::round(A[0][index]);
    return std::move(A);
  }
  // -------------------------------------------------------------------------------
  Matrix ceil(Matrix &A)
  {
    Matrix result(A.size(), 0);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::ceil(A[0][index]);
    return result;
  }

  Matrix &&ceil(Matrix &&A)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::round(A[0][index]);
    return std::move(A);
  }
  // -------------------------------------------------------------------------------
  Matrix mul(const Matrix &A, const Matrix &B)
  {
    if (A.size().N != B.size().M) {
      std::stringstream error_message;
      error_message << "error: mul: nonconformant arguments (";
      error_message << "A is " << A.size().M << 'x' << A.size().N;
      error_message << ", B is " << B.size().M << 'x' << B.size().N << ")\n";
      throw std::invalid_argument(error_message.str());
    }

    Matrix result(A.size().M, B.size().N, 0);
    for (size_t i = 0; i < B.size().N; i++)
      for (size_t j = 0; j < A.size().M; j++)
        for (size_t k = 0; k < A.size().N; k++)
          result[j][i] += A[j][k] * B[k][i];
    return result;
  }
  // -------------------------------------------------------------------------------
  Matrix transpose(const Matrix &A)
  {
    Matrix result(A.size().N, A.size().M);
    for (size_t y = 0; y < A.size().M; ++y)
      for (size_t x = 0; x < A.size().N; ++x)
        result[x][y] = A[y][x];
    return result;
  }

  Matrix reshape(const Matrix &A, const size_t M, const size_t N)
  {
    if (A.numel() != M * N) {
      std::stringstream error_message;
      error_message << "error: reshape: can't reshape " << A.size().M << 'x' << A.size().N;
      error_message << " array to " << M << 'x' << N << " array";
      throw std::invalid_argument(error_message.str());
    }
    Matrix result(M, N);
    for (size_t k = 0; k < result.numel(); ++k)
      result[0][k] = A[0][k];
    return result;
  }
  // -------------------------------------------------------------------------------
  double min(const Matrix &matrix)
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t k = 0; k < matrix.numel(); ++k)
      if (matrix[0][k] < minimum)
        minimum = matrix[0][k];
    return minimum;
  }

  double min(const Slice &column)
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t k = 0; k < column.numel(); ++k)
      if (column[k] < minimum)
        minimum = column[k];
    return minimum;
  }
  // -------------------------------------------------------------------------------
  double max(const Matrix &matrix)
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t k = 0; k < matrix.numel(); ++k)
      if (matrix[0][k] > maximum)
        maximum = matrix[0][k];
    return maximum;
  }

  double max(const Slice &column)
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t k = 0; k < column.numel(); ++k)
      if (column[k] > maximum)
        maximum = column[k];
    return maximum;
  }
  // -------------------------------------------------------------------------------
  double sum(const Matrix &A)
  {
    double summation = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      summation += A[0][k];
    return summation;
  }

  double prod(const Matrix &A)
  {
    double temporary = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      temporary += std::log(A[0][k]);
    return std::exp(temporary);
  }

  double avg(const Matrix& A)
  {
    const double iN = 1/A.numel();
    double average = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      average += A[0][k] * iN;
    return average;
  }

  double rms(const Matrix &A)
  {
    const size_t N = A.numel();
    double temporary = 0;
    for (size_t k = 0; k < N; ++k)
      temporary += A[0][k] * A[0][k];
    return std::sqrt(temporary);
  }

  double geo(const Matrix& A)
  {
    double temporary = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      temporary += std::log(A[0][k]);
    return std::exp(temporary / A.numel());
  }
  // -------------------------------------------------------------------------------
  Matrix orthogonalize(Matrix A)
  {
    const size_t M = A.size().M;
    const size_t N = A.size().N;

    Matrix Q(M, N);

    // orthogonalize A using the modified Gram-Schmidt process
    size_t i, j, k;
    double sum_of_squares, inorm, projection;
    for (i = 0; i < N; ++i) {
      sum_of_squares = 0;
      for (size_t j = 0; j < M; ++j)
        sum_of_squares += A[j][i] * A[j][i];
      inorm = std::pow(sum_of_squares, -0.5);
      if (std::isfinite(inorm))
        for (j = 0; j < M; ++j)
          Q[j][i] = A[j][i] * inorm;
      for (k = i + 1; k < N; ++k)
      {
        projection = 0;
        for (j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];
        for (j = 0; j < M; ++j)
          A[j][k] -= projection * Q[j][i];
      }
    }
    return Q;
  }

  std::unique_ptr<Matrix[]> QR(Matrix A)
  {
    const size_t M = A.size().M;
    const size_t N = A.size().N;

    std::unique_ptr<Matrix[]> QR(new Matrix[2]{{Matrix(M, N)}, Matrix(N, N, 0)});
    Matrix &Q = QR[0];
    Matrix &R = QR[1];

    size_t i, j, k;
    double sum_of_squares, inorm, projection;
    for (i = 0; i < N; ++i) {
      sum_of_squares = 0;
      for (size_t j = 0; j < M; ++j)
        sum_of_squares += A[j][i] * A[j][i];
      inorm = std::pow(sum_of_squares, -0.5);
      if (std::isfinite(inorm))
        for (j = 0; j < M; ++j)
          Q[j][i] = A[j][i] * inorm;
      for (k = i; k < N; ++k) {
        projection = 0;
        for (j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];
        if (k != i)
          for (j = 0; (k != i) && (j < M); ++j)
            A[j][k] -= projection * Q[j][i];
        if (k >= i)
          R[i][k] = projection;
      }
    }
    return QR;
  }

  Matrix div(const Matrix &b, Matrix A)
  {
    // verify vertical dimensions
    if (b.size().M != A.size().M) {
      std::stringstream error_message;
      error_message << "error: div: vertical dimensions mismatch (b is ";
      error_message << b.size().M << "x_, A is " << A.size().M << "x_)\n";
      throw std::invalid_argument(error_message.str());
    }

    // verify that b is a column matrix
    if (b.size().N != 1) {
      std::stringstream error_message;
      error_message << "error: div: horizontal dimension is not 1 (b is "
                    << "_x" << A.size().N << ")\n";
      throw std::invalid_argument(error_message.str());
    }

    // store the dimensions of A
    const size_t M = A.size().M;
    const size_t N = A.size().N;

    // necessary matrices
    Matrix Q(M, N), R(N, N), x(N, 1);

    // temporary variables
    double sum_of_squares, inorm, projection, substitution;
    size_t i, j, k;

    // reduced QR decomposition using the modified Gram-Schmidt process
    for (i = 0; i < N; ++i) {
      // calculate the squared Euclidean norm of A's i'th column
      sum_of_squares = 0;
      for (j = 0; j < M; ++j)
        sum_of_squares += A[j][i] * A[j][i];

      // skips if the squared Euclidean norm is 0
      if (sum_of_squares != 0) {
        // calculate the inverse Euclidean norm of A's i'th column
        inorm = std::pow(sum_of_squares, -0.5);
        // normalize and store A's normalized i'th column
        for (j = 0; j < M; ++j)
          Q[j][i] = A[j][i] * inorm;
      }

      // orthogonalize the remaining columns with respects to A's i'th column
      for (k = i; k < N; ++k) {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        projection = 0;
        for (j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];

        // construct upper triangle matrix R using Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k >= i)
          R[i][k] = projection;

        // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k != i) // skips if k == i because the projection would be 0
          for (j = 0; j < M; ++j)
            A[j][k] -= projection * Q[j][i];
      }
    }

    // solve linear system Rx = Qt*b using back substitution
    for (i = N - 1; i < N; --i) {
      // calculate appropriate Qt*b component
      substitution = 0;
      for (j = 0; j < M; ++j)
        substitution += Q[j][i] * b[j][0];

      // back substitution of previously solved x components
      for (k = N - 1; k > i; --k)
        substitution -= R[i][k] * x[k][0];

      // solve x's i'th component
      x[i][0] = substitution / R[i][i];
    }
    return x;
  }

  std::unique_ptr<Matrix[]> linearize(const Matrix &data_x, const Matrix &data_y)
  {
    // data is interpreted as a horizontal vector
    if ((data_x.size().M != 1) || (data_y.size().M != 1))
      std::clog << "warning: linearize: data is interpreted as a horizontal 1-dimensional matrix\n";
    // validate data set
    if (data_x.numel() != data_y.numel()) {
      std::stringstream error_message;
      error_message << "error: linearize: x and y data should have the same amount of elements";
      throw std::invalid_argument(error_message.str());
    }

    const size_t N = data_x.numel();

    std::unique_ptr<Matrix[]> data_set(new Matrix[2]{{Matrix(1, N, 0)}, {Matrix(1, N, 0)}});
    Matrix &lin_x = data_set[0];
    Matrix &lin_y = data_set[1];

    // set first and last values of the linearized data set
    lin_x[0][0] = data_x[0][0];
    lin_y[0][0] = data_y[0][0];
    lin_x[0][N - 1] = data_x[0][N - 1];
    lin_y[0][N - 1] = data_y[0][N - 1];

    // build linearly spaced x data and its associated y value
    const double step = (data_x[0][N - 1] - data_x[0][0]) / (N - 1);
    double x1, x2, y1, y2;
    for (size_t index = 1; index < (N - 1); ++index) {
      // build linearly spaced x data
      lin_x[0][index] = lin_x[0][index - 1] + step;

      // linearly interpolate y value
      x1 = data_x[0][index];
      y1 = data_y[0][index];
      x2 = data_x[0][index + 1];
      y2 = data_y[0][index + 1];
      lin_y[0][index] = y1 + (lin_x[0][index] - x1) * (y2 - y1) / (x2 - x1);
    }

    return data_set;
  }

  Matrix linspace(const double x1, const double x2, const size_t N)
  {
    Matrix vector(1, N);
    double step = (x2 - x1) / (N - 1);
    vector[0][0] = x1;
    vector[0][N - 1] = x2;
    for (size_t index = 1; index < (N - 1); ++index)
      vector[0][index] = vector[0][index - 1] + step;
    return vector;
  }

  Matrix linspace(const double x1, const double x2, const size_t N, Keyword::Column)
  {
    Matrix vector(N, 1);
    double step = (x2 - x1) / (N - 1);
    vector[0][0] = x1;
    vector[0][N - 1] = x2;
    for (size_t index = 1; index < (N - 1); ++index)
      vector[0][index] = vector[0][index - 1] + step;
    return vector;
  }

  Matrix iota(const size_t N)
  {
    Matrix indices(1, N);
    for (size_t index = 0; index < N; ++index)
      indices[0][index] = index;
    return indices;
  }

  Matrix diff(const Matrix &A, size_t n)
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

  Matrix reverse(const Matrix &A)
  {
    const size_t N = A.numel();
    Matrix result(A.size());
    for (size_t index = 0; index < N; ++index)
      result[0][index] = A[0][N-1 - index];
    return result;
  }

  Matrix&& reverse(Matrix &&A)
  {
    const size_t N   = A.numel();
    const size_t N_2 = N >> 1;
    for (size_t k = 0; k < N_2; ++k)
      std::swap(A[0][k], A[0][N-1 - k]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix conv(const Matrix &A, const Matrix &B)
  {
    const size_t n1 = A.numel();
    const size_t n2 = B.numel();
    Matrix convoluted(1, n1 + n2 - 1, 0);
    for (size_t i = 0; i < n1; ++i)
      for (size_t j = 0; j < n2; ++j)
        convoluted[0][i + j] += A[0][i] * B[0][j];
    return convoluted;
  }

  Matrix corr(const Matrix &A, const Matrix &B)
  {
    const size_t n1 = A.numel();
    const size_t n2 = B.numel();
    Matrix result(1, n1 + n2 - 1, 0);
    for (size_t i = 0; i < n1; ++i)
      for (size_t j = 0; j < n2; ++j)
        result[0][i + j] += A[0][n1-1 - i] * B[0][j];
    return result; 
  }

  Matrix corr(const Matrix &A)
  {
    const size_t n = A.numel();
    Matrix result(1, 2*n - 1, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        result[0][i + j] += A[0][n-1 - i] * A[0][j];
    return result;
  }

  Matrix Rxx(const Matrix &A)
  {
    const size_t n = A.numel();
    Matrix Rxx(1, n, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if ((i+j - n+1) < n)
          Rxx[0][i+j - n+1] += A[0][n-1 - i] * A[0][j];
    return Rxx;
  }

  Matrix Rxx(const Matrix &A, const size_t K)
  {
    const size_t n = A.numel();
    Matrix Rxx(1, K, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if ((i+j - n+1) < K)
          Rxx[0][i+j - n+1] += A[0][n-1 - i] * A[0][j];
    return Rxx;
  }

  Matrix lpc(const Matrix &A, const size_t p)
  {
    if (A.size().M != 1)
      std::clog << "warning: lpc: A should be a horizontal vector\n";
    if (p >= A.numel())
      throw std::invalid_argument("error: lpc: p should be smaller than the smaller of elements in A");

    Matrix rxx = Rxx(A, p+1);

    Matrix autocorr_mat(p, p);
    Matrix autocorr_vec(p, 1);
    for (size_t i = 0; i < p; ++i) {
      for (size_t j = 0; j < p; ++j)
        autocorr_mat[j][i] = rxx[0][(j > i) ? (j - i) : (i - j)];
      autocorr_vec[0][i] = rxx[0][i+1];
    }

    return div(autocorr_vec, autocorr_mat);
  }

  Matrix toeplitz(const Matrix& A)
  {
    const size_t n = A.numel();
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        result[j][i] = A[0][(j > i) ? (j - i) : (i - j)];

    return result;
  }
// -------------------------------------------------------------------------------
  Matrix blackman(const size_t N)
  {
    Matrix window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1)) + 0.08 * std::cos(4 * M_PI * k / (N - 1));
    return window;
  }

  Matrix blackman(const Matrix &signal)
  {
    const size_t N = signal.numel();
    Matrix windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1)) + 0.08 * std::cos(4 * M_PI * k / (N - 1)));
    return windowed;
  }

  Matrix &&blackman(Matrix &&signal)
  {
    const size_t N = signal.numel();
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1)) + 0.08 * std::cos(4 * M_PI * k / (N - 1));
    return std::move(signal);
  }
  // -------------------------------------------------------------------------------
  Matrix hamming(const size_t N)
  {
    Matrix window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.54 - 0.46 * std::cos(2 * M_PI * k / (N - 1));
    return window;
  }

  Matrix hamming(const Matrix &signal)
  {
    const size_t N = signal.numel();
    Matrix windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.54 - 0.46 * std::cos(2 * M_PI * k / (N - 1)));
    return windowed;
  }

  Matrix &&hamming(Matrix &&signal)
  {
    const size_t N = signal.numel();
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.54 - 0.46 * std::cos(2 * M_PI * k / (N - 1));
    return std::move(signal);
  }
  // -------------------------------------------------------------------------------
  Matrix hann(const size_t N)
  {
    Matrix window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.5 - 0.5 * cos(2 * M_PI * k / (N - 1));
    return window;
  }

  Matrix hann(const Matrix &signal)
  {
    const size_t N = signal.numel();
    Matrix windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.5 - 0.5 * cos(2 * M_PI * k / (N - 1)));
    return windowed;
  }

  Matrix &&hann(Matrix &&signal)
  {
    const size_t N = signal.numel();
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.5 - 0.5 * cos(2 * M_PI * k / (N - 1));
    return std::move(signal);
  }
  // -------------------------------------------------------------------------------
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed)
  {
    const double half_tol = tol * 0.5;

    double root = seed;

    size_t iteration = 0;
    while ((tol < std::abs(function(root))) && (iteration++ < max_iteration))
      root -= tol * function(root) / (function(root + half_tol) - function(root - half_tol));

    return root;
  }

  Matrix sin(Matrix &A)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::sin(A[0][index]);
    return result;
  }

  Matrix &&sin(Matrix &&A)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::sin(A[0][index]);
    return std::move(A);
  }

  double sinc(const double x)
  {
    return x == 0 ? 1 : std::sin(M_PI * x) / (M_PI * x);
  }

  Matrix sinc(Matrix &A)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = sinc(A[0][index]);
    return result;
  }

  Matrix &&sinc(Matrix &&A)
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = sinc(A[0][index]);
    return std::move(A);
  }

  Matrix upsample(const Matrix &data, const size_t L)
  {
    Matrix upsampled(1, L * data.numel(), 0);
    for (size_t index = 0; index < data.numel(); ++index)
      upsampled[0][index * L] = data[0][index];
    return upsampled;
  }

  Matrix sinc_impulse(const size_t length, const double frequency)
  {
    // validate impulse length
    if (!(length % 2))
      throw std::invalid_argument("error: sinc_impulse: impulse length must be odd");

    // offset to the impulse center
    const signed offset = (length - 1) * 0.5;

    // compute impulse
    Matrix impulse(1, length);
    for (signed k = 0; k < signed(length); ++k)
      impulse[0][k] = sinc((k - offset) * frequency);

    return impulse;
  }

  Matrix resample(const Matrix &data, const size_t L)
  {
    if (data.size().M != 1)
      std::clog << "warning: resample: data is interpreted as a horizontal 1-dimensional matrix\n";
    if (data.numel() == 0)
      throw std::invalid_argument("error: resample: data must contain atleast 1 element");

    const size_t N = data.numel();
    // offset to impulse center
    const size_t offset = L * 3.5;
    // length of impulse
    const size_t length = 2 * offset + 1;
    // indices to the first and last upsampled data elements in the symetrically extended data
    const size_t first = L * 2;
    const size_t last = L * (N + 1);

    // temporary variables
    size_t i, j, k;

    // symetrically extended data vector
    Matrix extended(1, (N + 4) * L, 0);
    // store and upsample left symetrical data
    extended[0][0] = 2 * data[0][0] - data[0][2];
    extended[0][L] = 2 * data[0][0] - data[0][1];
    // store and upsample right symetrical data
    extended[0][last + L] = 2 * data[0][N - 1] - data[0][N - 2];
    extended[0][last + L * 2] = 2 * data[0][N - 1] - data[0][N - 3];
    // store and upsample data
    for (k = 0; k < N; ++k)
      extended[0][L * k + first] = data[0][k];

    // design low-pass interpolation filter
    const Matrix filter = blackman(sinc_impulse(length, 1.0 / L));

    // interpolate upsampled data using a cropped convolution
    Matrix resampled(1, last - first + 1, 0);
    for (i = 0; i < extended.numel(); i += L) {
      for (j = 0; j < filter.numel(); ++j) {
        k = i + j - offset;
        // skips if the index is not within the upsampled data range
        if ((first <= k) && (k <= last))
          resampled[0][k - first] += extended[0][i] * filter[0][j];
      }
    }

    return resampled;
  }

  Matrix resample(const Matrix &data, const size_t L, const size_t keep, const double alpha, const bool tail)
  {
    if (data.size().M != 1)
      std::clog << "warning: resample: data is interpreted as a horizontal 1-dimensional matrix\n";
    if (data.numel() == 0)
      throw std::invalid_argument("error: resample: data must contain atleast 1 element");
    if (L <= 1)
      throw std::invalid_argument("error: resample: L must be 2 or more");
    if (keep >= data.numel())
      throw std::invalid_argument("error: resample: keep must be less than the number of elements of data");
    if (alpha < 1)
      throw std::invalid_argument("error: resample: alpha must be at least 1/L");
    const size_t N = data.numel();
    // offset to impulse center
    const size_t offset = L * alpha;
    // length of impulse
    const size_t length = 2 * offset + 1;

    // temporary variables
    size_t i, j, k;

    // symetrically extended data vector
    Matrix extended(1, (N + 2 * keep) * L, 0);
    // store and upsample left symetrical data
    k = 0;
    for (i = 0; i < keep; ++i) {
      extended[0][k] = 2 * data[0][0] - data[0][keep - i];
      k += L;
    }
    // index to the first upsampled element in the symetrically extended data
    const size_t first = k;
    // store and upsample data
    for (i = 0; i < N; ++i) {
      extended[0][k] = data[0][i];
      k += L;
    }
    // index to the last upsampled element in the symetrically extended data
    const size_t last = k - (tail ? 1 : L);
    // store and upsample right symetrical data
    for (i = 0; i < keep; ++i) {
      extended[0][k] = 2 * data[0][N - 1] - data[0][N - 2 - i];
      k += L;
    }

    // design low-pass interpolation filter
    const Matrix filter = blackman(sinc_impulse(length, 1.0 / L));

    // interpolate upsampled data using a cropped convolution
    Matrix resampled(1, last - first + 1, 0);
    for (i = 0; i < extended.numel(); ++i) {
      for (j = 0; j < filter.numel(); ++j) {
        k = i + j - offset;
        // skips if the index is not within the upsampled data range
        if ((first <= k) && (k <= last))
          resampled[0][k - first] += extended[0][i] * filter[0][j];
      }
    }

    return resampled;
  }

  std::ostream &operator<<(std::ostream &ostream, const Matrix &A)
  {
    if (A.numel()) {
      std::size_t max_len = 0;

      for (size_t y = 0; y < A.size().M; ++y) {
        for (size_t x = 0; x < A.size().N; ++x) {
          std::stringstream ss;
          ss.copyfmt(ostream);
          ss << A[y][x];
          max_len = std::max(max_len, ss.str().length());
        }
      }

      for (size_t y = 0; y < A.size().M; ++y) {
        for (size_t x = 0; x < A.size().N; ++x)
          ostream << std::setw(max_len + 1) << A[y][x];
        ostream << '\n';
      }
    }
    return ostream;
  }

  std::ostream &operator<<(std::ostream &ostream, const Slice &A)
  {
    if (A.numel()) {
      std::size_t max_len = 0;
      for (size_t y = 0; y < A.numel(); ++y) {
        std::stringstream ss;
        ss.copyfmt(ostream);
        ss << A[y];
        max_len = std::max(max_len, ss.str().length());
      }
      if (A.size().M == 1) {
        ostream << std::setw(max_len) << A[0];
        for (size_t x = 1; x < A.numel(); ++x)
          ostream << ' ' << std::setw(max_len) << A[x];
        ostream << '\n';
      }
      else {
        for (size_t y = 0; y < A.numel(); ++y)
          ostream << std::setw(max_len) << A[y] << '\n';
      }
    }
    return ostream;
  }

  std::ostream &operator<<(std::ostream &ostream, const Size size)
  {
    return ostream << size.M << 'x' << size.N;
  }

  void plot(std::string title, List<DataSet> data_sets, bool persistent, bool remove, bool pause, bool lines)
  {
    // validate that gnuplot is in system path
    static bool gnuplot_on_system_path = false;
    if ((!gnuplot_on_system_path) && std::system("gnuplot --version"))
      throw std::runtime_error("error: plot: gnuplot could not be found in the system path");
    gnuplot_on_system_path = true;

    // validate that x and y have the same number of elements
    for (auto &data_set : data_sets) {
      const Matrix &xdata = data_set.first;
      const Matrix &ydata = data_set.second;
      if (xdata.numel() != ydata.numel()) {
        std::stringstream error_message;
        error_message << "error: plot: number of element mismatch (x has " << xdata.numel();
        error_message << " elements,  y has " << ydata.numel() << " elements)\n";
        throw std::invalid_argument(error_message.str());
      }
    }

    // create temporary file
    std::ofstream file("gnuplot.data");

    // validate file opening
    if (!file)
      throw std::runtime_error("could not open gnuplot.data");

    // write x and y data to file for each data set
    for (auto &data_set : data_sets) {
      const Matrix &x = data_set.first;
      const Matrix &y = data_set.second;
      for (size_t index = 0; index < x.numel(); ++index)
        file << x[0][index] << ' ' << y[0][index] << '\n';
      // separate data sets
      file << "\n\n";
    }

    // close file
    file.close();

    // gnuplot command pipeline
    std::stringstream gnuplot_pipeline;
    gnuplot_pipeline << "gnuplot";

    // conditionally set plot to persistent
    if (persistent)
      gnuplot_pipeline << " -persistent";

    // plot all data sets
    gnuplot_pipeline << " -e \"set title \\\"gnuplot\\\"; plot 'gnuplot.data'";

    for (size_t k = 0; k < data_sets.size(); ++k) {
      if (k)
        gnuplot_pipeline << ", ''";
      gnuplot_pipeline << " index " << k;
      if (lines)
        gnuplot_pipeline << " with lines";
      gnuplot_pipeline << " title '" << title << '\'';
    }
    gnuplot_pipeline << '"';

    // conditionally pause after plotting
    if (pause)
      gnuplot_pipeline << " -e \"pause -1 'press any key to continue...'\"";

    // execute command pipeline
    std::system(gnuplot_pipeline.str().c_str());

    // conditionally remove file after creation
    if (remove)
      std::remove("gnuplot.data");
  }

  void plot(List<std::string> titles, List<DataSet> data_sets, bool persistent, bool remove, bool pause, bool lines)
  {
    // validate that gnuplot is in system path
    static bool gnuplot_on_system_path = false;
    if ((!gnuplot_on_system_path) && std::system("gnuplot --version"))
      throw std::runtime_error("error: plot: gnuplot could not be found in the system path");
    gnuplot_on_system_path = true;

    // validate that x and y have the same number of elements
    for (auto &data_set : data_sets) {
      const Matrix &xdata = data_set.first;
      const Matrix &ydata = data_set.second;
      if (xdata.numel() != ydata.numel()) {
        std::stringstream error_message;
        error_message << "error: plot: number of element mismatch (x has " << xdata.numel();
        error_message << " elements,  y has " << ydata.numel() << " elements)\n";
        throw std::invalid_argument(error_message.str());
      }
    }

    if (titles.size() != data_sets.size())
      std::clog << "warning: plot: number of titles does not equal number of data sets\n";

    // create temporary file
    std::ofstream file("gnuplot.data");

    // validate file opening
    if (!file)
      throw std::runtime_error("could not open gnuplot.data");

    // write x and y data to file for each data set
    for (auto &data_set : data_sets) {
      const Matrix &x = data_set.first;
      const Matrix &y = data_set.second;
      for (size_t index = 0; index < x.numel(); ++index)
        file << x[0][index] << ' ' << y[0][index] << '\n';
      // separate data sets
      file << "\n\n";
    }

    // close file
    file.close();

    // gnuplot command pipeline
    std::stringstream gnuplot_pipeline;
    gnuplot_pipeline << "gnuplot";

    // conditionally set plot to persistent
    if (persistent)
      gnuplot_pipeline << " -persistent";

    // plot all data sets
    gnuplot_pipeline << " -e \"set title \\\"gnuplot\\\"; plot 'gnuplot.data'";
    size_t k = 0;
    for (std::string title : titles) {
      if (k)
        gnuplot_pipeline << ", ''";
      gnuplot_pipeline << " index " << k;
      if (lines)
        gnuplot_pipeline << " with lines";
      gnuplot_pipeline << " title '" << title << '\'';
      k++;
    }
    gnuplot_pipeline << '"';

    // conditionally pause after plotting
    if (pause)
      gnuplot_pipeline << " -e \"pause -1 'press any key to continue...'\"";

    // execute command pipeline
    std::system(gnuplot_pipeline.str().c_str());

    // conditionally remove file after creation
    if (remove)
      std::remove("gnuplot.data");
  }

  void plot(std::string title, DataSet data_set, bool persistent, bool remove, bool pause, bool lines)
  {
    plot(title, {data_set}, persistent, remove, pause, lines);
  }
}}
//
int main()
{
  using namespace Pinakas;
  using namespace Keyword;

  /*
  auto   f = [](const Matrix &x) {return ((1-x)^ 2) + sin(x * 5) / 5 - 2;};
  size_t N = 10;
  size_t L = 100;

  Matrix x_nonlinear = linspace(0, 1, N) + Random(0, 1.0/(N-1));
  Matrix y_nonlinear = f(x_nonlinear);

  Matrix y_nonlinear_resampled = resample(y_nonlinear, L);
  Matrix x_nonlinear_resampled = resample(x_nonlinear, L);

  auto xy_linearized = linearize(x_nonlinear, y_nonlinear);
  Matrix y_linearized_resampled  = resample(xy_linearized[1], L);
  Matrix x_linearized_resampled  = linspace(x_nonlinear(0), x_nonlinear(end), y_linearized_resampled.numel());

  Matrix x_linear = linspace(x_nonlinear(0), x_nonlinear(end), N);
  Matrix y_linear = f(x_linear);

  Matrix y_linear_resampled = resample(y_linear, L);
  Matrix x_linear_resampled = linspace(x_nonlinear(0), x_nonlinear(end), y_linear_resampled.numel());

  Matrix x_true = linspace(x_nonlinear(0), x_nonlinear(end), N*L);
  Matrix y_true = f(x_true);

  plot({"truth", "linear resampled", "non-linear resampled", "linearized resampled"},
                                        {{x_true, y_true},
                                        {x_linear_resampled, y_linear_resampled},
                                        {x_nonlinear_resampled, y_nonlinear_resampled},
                                        {x_linearized_resampled, y_linearized_resampled}});
  //*/
  
  
  size_t N = 1000;
  size_t L = 1000;

  Matrix signal = linspace(0, 1, N) + Random(0, 1.0/(N-1));

  for (int i = 0; i < 0; ++i) {
    {
      Chronometro::Stopwatch sw;
      resample(signal, L);
    }
    puts("-----------------");
  }
  plot({"resampling"}, {{iota(N*L-L+1), resample(signal, L)}});

  /*
  Matrix time   = linspace(0, 1, 100);
  Matrix signal = (time^2);
  signal(30) = 1.5;
  signal(31) = 1.7;
  signal(32) = 1.9;
  signal(33) = 1.9;
  signal(34) = 1.7;
  signal(35) = 1.5;
  Matrix y = Rxx(signal, 20);
  Matrix x = linspace(0, 1, y.numel());
  Matrix lpc_coeffs = lpc(signal, 5);
  std::cout << lpc_coeffs;
  plot({"signal"}, {{time, signal}});
  //*/
}