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
  
  Matrix::Matrix(const Matrix& other)
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
  
  Matrix::Matrix(Matrix&& other)
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
    : Matrix(size.M, size.N)
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
  
  Matrix::Matrix(const size_t M, const size_t N, const Range range)
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
  
  Matrix::Matrix(const Size size, const Range range)
    : Matrix(size.M, size.N, range)
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
    for (const List<const double>& vector : values) {
      if (temp_N && (temp_N != vector.size())) {
        std::cerr << "vertical dimensions mismatch (" << temp_N << " vs " << vector.size() << ")\n";
        size_ = {0, 0, 0};
        return;
      }
      else temp_N = vector.size();
    }

    // allocate memory
    allocate(this, values.size(), temp_N);

    // store values
    size_t y = 0;
    for (const List<const double>& vector : values) {
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
    for (const Matrix& matrix : list) {
      if (temp_M && (temp_M != matrix.size_.M)) {
        std::cerr << "vertical dimensions mismatch (" << temp_M << " vs " << matrix.size_.M << ")\n";
        size_ = {0, 0, 0};
        return;
      }
      else temp_M = matrix.size_.M;
      temp_N += matrix.size_.N;
    }

    // allocate memory
    allocate(this, temp_M, temp_N);

    // store values
    size_t index = 0;
    for (const Matrix& matrix : list) {
      for (size_t y = 0; y < matrix.size_.M; ++y)
        for (size_t x = 0; x < matrix.size_.N; ++x)
          data_[x + index + y * size_.N] = matrix[y][x];
      index += matrix.size_.N;
    }
  }

  void allocate(Matrix* matrix, const size_t M, const size_t N)
  {
    // validate sizes
    if ((M == 0) || (N == 0)) {
      std::stringstream error_message;
      error_message << "invalid didimensions are " << M << 'x' << N;
      throw std::invalid_argument(error_message.str());
    }

    // allocate memory
    matrix->data_.reset((double*) new char[sizeof(double[M][N])]);

    // validate memory allocation
    if (!matrix->data_.get())
      throw std::bad_alloc();

    // save size information
    matrix->size_ = {M, N, M*N};
  }
 
  double* Matrix::operator[](const size_t index) const
  {
    return data_.get() + index*size_.N;
  }
 
  double& Matrix::operator()(const size_t index) const
  {
    if (index >= size_.numel) {
      std::stringstream error_message;
      error_message << '(' << index << ") out of bound " << size_.numel - 1<< " (dimensions are " << size_ << ')';
      throw std::out_of_range(error_message.str());
    }
    return data_[index];
  }

  double& Matrix::operator()(Keyword::End) const
  {
    return data_[size_.numel - 1];
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
    return data_[x + y*size_.N];
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
// -------------------------------------------------------------------------------
  template<typename T>
  Iterator<T>::Iterator(T& matrix, const size_t index)
    : // member initialization list
    matrix(matrix),
    index(index)
  {}

  template<typename T>
  bool Iterator<T>::operator==(const Iterator<T>& other) const
  {
    return index == other.index;
  }

  template<typename T>
  bool Iterator<T>::operator!=(const Iterator<T>& other) const
  {
    return this->index != other.index;
  }

  template<typename T>
  Iterator<T>& Iterator<T>::operator++(void)
  {
    ++index;
    return *this;
  }

  template<typename T>
  double& Iterator<T>::operator*(void) const
  {
    return matrix[0][index];
  }
// -------------------------------------------------------------------------------
  template<typename T>
  ConstIterator<T>::ConstIterator(const T& matrix, const size_t index)
    : matrix(matrix), index(index)
  {}

  template<typename T>
  bool ConstIterator<T>::operator==(const ConstIterator<T>& other) const
  {
    return index == other.index;
  }

  template<typename T>
  bool ConstIterator<T>::operator!=(const ConstIterator<T>& other) const
  {
    return this->index != other.index;
  }

  template<typename T>
  ConstIterator<T>& ConstIterator<T>::operator++()
  {
    ++index;
    return *this;
  }
  
  template<typename T>
  const double& ConstIterator<T>::operator*(void) const
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

  ConstIterator<Matrix> Matrix::begin(void) const {
    return ConstIterator<Matrix>(*this, 0);
  }

  ConstIterator<Matrix> Matrix::end(void) const
  {
    return ConstIterator<Matrix>(*this, size_.numel);
  }
// -------------------------------------------------------------------------------
  Matrix& Matrix::operator=(const Matrix& other) &
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

  Matrix& Matrix::operator=(Matrix&& other) &
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
  
  Matrix& Matrix::operator=(const double value) &
  {
    // store values
    for (size_t index = 0; index < size_.numel; ++index)
      data_[index] = value;
    return *this;
  }
// -------------------------------------------------------------------------------
  Slice::Slice(Matrix& matrix, const size_t n, Keyword::Column)
    : // member initialization list,
    size_{matrix.size().M, 1, matrix.size().M},
    fixed_(n),
    col_row_(false),
    matrix_(matrix)
  {}

  Slice::Slice(Matrix& matrix, const size_t m, Keyword::Row)
    : // member initialization list,
    size_{1, matrix.size().N, matrix.size().N},
    fixed_(m),
    col_row_(true),
    matrix_(matrix)
  {}

  double& Slice::operator[](const size_t index) const
  {
    return col_row_ ? matrix_[fixed_][index] : matrix_[index][fixed_];
  }

  double& Slice::operator()(const size_t index) const
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

  Size Slice::size(void) const
  {
    return size_;
  }
// -------------------------------------------------------------------------------
  Range::Range(double min, double max)
    : // member initialization list
    min_(min),
    max_(max)
  {}
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
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index] + B[0][index];
    return res;
  }
  
  Matrix operator+(const Matrix& A, const Range range)
  {
    Matrix res(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] += A[0][index] + uniform_distribution(generator);
    return res;
  }
  
  Matrix&& operator+(Matrix&& A, const Range range)
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += uniform_distribution(generator);
    return std::move(A);
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
  
  Matrix& operator+=(Matrix& A, const double B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += B;
    return A;
  }
  
  Matrix operator+(const Matrix& A, const double B)
  {
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index] + B;
    return res;
  }
  
  Matrix&& operator+(Matrix&& A, const double B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] += B;
    return std::move(A += B);
  }
  
  Matrix operator+(const double A, const Matrix& B)
  {
    Matrix res(B.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A + B[0][index];
    return res;
  }
  
  Matrix&& operator+(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] += A;
    return std::move(B);
  }
  
  Matrix operator+(const Matrix& A)
  {
    return A;
  }
  
  Matrix&& operator+(Matrix&& A)
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
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index] * B[0][index];
    return res;
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
  
  Matrix operator*(const Matrix& A, const double B)
  {
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index] * B;
    return res;
  }
  
  Matrix&& operator*(Matrix&& A, const double B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] *= B;
    return std::move(A *= B);
  }
  
  Matrix operator*(const double A, const Matrix& B)
  {
    Matrix res(B.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A * B[0][index];
    return res;
  }
  
  Matrix&& operator*(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] *= A;
    return std::move(B);
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
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index] - B[0][index];
    return res;
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

  Matrix& operator-=(Matrix& A, const double B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] -= B;
    return A;
  }

  Matrix operator-(const Matrix& A, const double B)
  {
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index] - B;
    return res;
  }
  
  Matrix&& operator-(Matrix&& A, const double B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] -= B;
    return std::move(A);
  }

  Matrix operator-(const double A, const Matrix& B)
  {
    Matrix res(B.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A - B[0][index];
    return res;
  }

  Matrix&& operator-(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] = A - B[0][index];
    return std::move(B);
  }

  Matrix operator-(const Matrix& A)
  {
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = -A[0][index];
    return res;
  }

  Matrix&& operator-(Matrix&& A)
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
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index] / B[0][index];
    return res;
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
  Matrix& operator/=(Matrix& A, const double B)
  {
    const double iB = 1 / B;
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] *= iB;
    return A;
  }
  
  Matrix operator/(const Matrix& A, const double B)
  {
    const double iB = 1 / B;
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index] * iB;
    return res;
  }
  
  Matrix&& operator/(Matrix&& A, const double B)
  {
    const double iB = 1 / B;
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] *= iB;
    return std::move(A);
  }

  Matrix operator/(const double A, const Matrix& B)
  {
    Matrix res(B.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A / B[0][index];
    return res;
  }

  Matrix&& operator/(const double A, Matrix&& B)
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
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = std::pow(A[0][index], B[0][index]);
    return res;
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
  Matrix& operator^=(Matrix& A, const double B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::pow(A[0][index], B);
    return A;
  }

  Matrix operator^(const Matrix& A, const double B)
  {
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = std::pow(A[0][index], B);
    return res;
  }
  
  Matrix&& operator^(Matrix&& A, const double B)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::pow(A[0][index], B);
    return std::move(A);
  }

  Matrix operator^(const double A, const Matrix& B)
  {
    Matrix res(B.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = std::pow(A, B[0][index]);
    return res;
  }

  Matrix&& operator^(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.size().numel; ++index)
      B[0][index] = std::pow(A, B[0][index]);
    return std::move(B);
  }
// -------------------------------------------------------------------------------
  Matrix floor(const Matrix& A)
  {
    Matrix res(A.size(), 0);
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = std::floor(A[0][index]);
    return res;
  }

  Matrix&& floor(Matrix&& A)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::floor(A[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix round(const Matrix& A)
  {
    Matrix res(A.size(), 0);
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = std::round(A[0][index]);
    return res;
  }
  
  Matrix&& round(Matrix&& A)
  {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = std::round(A[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix ceil(Matrix& A)
  {
    Matrix res(A.size(), 0);
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = std::ceil(A[0][index]);
    return res;
  }
  
  Matrix&& ceil(Matrix&& A)
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
      error_message << "operator mul: nonconformant arguments (";
      error_message << "A is " << A.size().M << 'x' << A.size().N;
      error_message << ", B is " << B.size().M << 'x' << B.size().N << ")\n";
      throw std::invalid_argument(error_message.str());
    }
    Matrix res(A.size().M, B.size().N, 0);
    for (size_t i = 0; i < B.size().N; i++)
      for (size_t j = 0; j < A.size().M; j++)
        for (size_t k = 0; k < A.size().N; k++)
          res[j][i] += A[j][k] * B[k][i];
    return res;
  }
// -------------------------------------------------------------------------------
  Matrix transpose(const Matrix& A)
  {
    Matrix res(A.size().N, A.size().M);
    for (size_t y = 0; y < A.size().M; ++y)
      for (size_t x = 0; x < A.size().N; ++x)
        res[x][y] = A[y][x];
    return res;
  }

  Matrix reshape(const Matrix& A, const size_t M, const size_t N)
  {
    if (A.size().numel != M*N) {
      std::stringstream error_message;
      error_message << "reshape: can't reshape " << A.size();
      error_message << " array to " << M << 'x' << N << " array";
      throw std::invalid_argument(error_message.str());
    }
    Matrix res(M, N);
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = A[0][index];
    return res;
  }
// -------------------------------------------------------------------------------
  double min(const Matrix& matrix)
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t index = 0; index < matrix.size().numel; ++index)
      if (matrix[0][index] < minimum)
        minimum = matrix[0][index];
    return minimum;
  }

  double min(const Slice& column)
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t index = 0; index < column.size().numel; ++index)
      if (column[index] < minimum)
        minimum = column[index];
    return minimum;
  }
// -------------------------------------------------------------------------------
  double max(const Matrix& matrix)
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t index = 0; index < matrix.size().numel; ++index)
      if (matrix[0][index] > maximum)
        maximum = matrix[0][index];
    return maximum;
  }

  double max(const Slice& column)
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t index = 0; index < column.size().numel; ++index)
      if (column[index] > maximum)
        maximum = column[index];
    return maximum;
  }
// -------------------------------------------------------------------------------
  double sum(const Matrix& matrix)
  {
    double summation = 0;
    for (size_t index = 0; index < matrix.size().numel; ++index)
      summation += matrix[0][index];
    return summation;
  }
// -------------------------------------------------------------------------------
  double prod(const Matrix& matrix)
  {
    double product = 0;
    for (size_t index = 0; index < matrix.size().numel; ++index)
      product *= matrix[0][index];
    return product;
  }
// -------------------------------------------------------------------------------
  Matrix MGS(Matrix A)
  {
    const size_t M = A.size().M;
    const size_t N = A.size().N;

    Matrix Q(M, N);
    
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
      for (k = i + 1; k < N; ++k) {
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
    Matrix& Q = QR[0];
    Matrix& R = QR[1];
    
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

  Matrix div(const Matrix& b, Matrix A)
  {
    // verify vertical dimensions
    if (b.size().M != A.size().M) {
      std::stringstream error_message;
      error_message << "operator div: vertical dimensions mismatch (b is ";
      error_message << b.size().M << "x_, A is " << A.size().M << "x_)\n";
      throw std::invalid_argument(error_message.str());
    }

    // verify that b is a column matrix
    if (b.size().N != 1) {
      std::stringstream error_message;
      error_message << "operator div: horizontal dimension is not 1 (b is " << "_x" << A.size().N << ")\n";
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
  
  std::pair<Matrix, Matrix> linearize(const DataSet data_set)
  {
    // validate data set
    if (data_set.first.size() != data_set.second.size()) {
      std::stringstream error_message;
      error_message << "";
      throw std::invalid_argument(error_message.str());
    }

    const Matrix& data_x = data_set.first;
    const Matrix& data_y = data_set.second;
    const size_t N = data_x.size().numel;
    const double step = (data_x[0][N-1] - data_x[0][0]) / (N - 1);
    Matrix lin_x(data_x.size(), 0);
    Matrix lin_y(data_y.size(), 0);

    // set starting and ending value of linearized data set
    lin_x[0][0] = data_x[0][0];
    lin_y[0][0] = data_y[0][0];
    lin_x[0][N - 1] = data_x[0][N - 1];
    lin_y[0][N - 1] = data_y[0][N - 1];

    // build linearly spaced x data and its associated y value
    double x1, x2, y1, y2;
    for (size_t index = 1; index < (N - 1); ++index) {
      // build linearly spaced x data
      lin_x[0][index] = lin_x[0][index-1] + step;

      // interpolate new y value for new linearly spaced x data
      x1 = data_x[0][index];
      y1 = data_y[0][index];
      x2 = data_x[0][index + 1];
      y2 = data_y[0][index + 1];
      lin_y[0][index] = ((y1 - y2) * lin_x[0][index] + x1 * y2 - x2 * y1) / (x1 - x2);
    }
    return {lin_x, lin_y};
  }

  Matrix linspace(const double x1, const double x2, const size_t N, Keyword::Row)
  {
    Matrix vector(1, N);
    double step = (x2 - x1) / (N - 1);
    vector[0][0] = x1;
    vector[0][N - 1] = x2;
    for (size_t index = 1; index < (N - 1); ++index)
      vector[0][index] = vector[0][index-1] + step;
    return vector;
  }

  Matrix linspace(const double x1, const double x2, const size_t N, Keyword::Column)
  {
    Matrix vector(N, 1);
    double step = (x2 - x1) / (N - 1);
    vector[0][0]     = x1;
    vector[0][N - 1] = x2;
    for (size_t index = 1; index < (N - 1); ++index)
      vector[0][index] = vector[0][index-1] + step;
    return vector;
  }

  Matrix iota(const size_t N)
  {
    Matrix indices(1, N);
    for (size_t index = 0; index < N ; ++index)
      indices[0][index] = index;
    return indices;
  }

  Matrix diff(const Matrix& A, Keyword::Row, size_t n)
  {
    if (n) {
      Matrix derivative(A.size().M, A.size().N - 1, 0);
      for (size_t y = 0; y < derivative.size().M; ++y)
        for (size_t x = 0; x < derivative.size().N; ++x)
          derivative[y][x] = A[y][x + 1] - A[y][x];

      return diff(derivative, Keyword::row, n - 1);
    }
    return A;
  }

  Matrix diff(const Matrix& A, Keyword::Column, size_t n)
  {
    if (n) {
      Matrix derivative(A.size().M - 1, A.size().N, 0);
      for (size_t y = 0; y < derivative.size().M; ++y)
        for (size_t x = 0; x < derivative.size().N; ++x)
          derivative[y][x] = A[y+ 1][x] - A[y][x];

      return diff(derivative, Keyword::column, n - 1);
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
    Matrix res(A.size());
    for (size_t index = 0; index < res.size().numel; ++index)
      res[0][index] = sinc(A[0][index], freq);
    return res;
  }

  Matrix&& sinc(Matrix&& A, double freq = 1) {
    for (size_t index = 0; index < A.size().numel; ++index)
      A[0][index] = sinc(A[0][index], freq);
    return std::move(A);
  }

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
    Matrix res(1, N-L+1); // cropped convolution of upsampled data with windowed filter impulse
    for (int i = 0; i<N-L+1; i++)
      for (int n = 0; n<l; n++)
        res[0][i] += u[0][(i+n-o) < N ? std::abs(i+n-o) : (2*N-L - (i+n-o))] * hw[0][n];

    return res;
  }

  std::ostream& operator<<(std::ostream& ostream, const Matrix& A)
  {
    if (A.size().numel) {
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
          ostream << std::setw(max_len+1) << A[y][x];
        ostream << '\n';
      }
    }
    return ostream;
  }

  std::ostream& operator<<(std::ostream& ostream, const Slice& A)
  {
    if (A.size().numel) {
      std::size_t max_len = 0;
      for (size_t y = 0; y < A.size().numel; ++y) {
        std::stringstream ss;
        ss.copyfmt(ostream);
        ss << A[y];
        max_len = std::max(max_len, ss.str().length());
      }
      if (A.size().M == 1) {
        ostream << std::setw(max_len) << A[0];
        for (size_t x = 1; x < A.size().numel; ++x)
            ostream << ' ' << std::setw(max_len) << A[x];
        ostream << '\n';
      }
      else {
        for (size_t y = 0; y < A.size().numel; ++y)
          ostream << std::setw(max_len) << A[y] << '\n';
      }
    }
    return ostream;
  }

  std::ostream& operator<<(std::ostream& ostream, const Size size)
  {
    return ostream << size.N << 'x' << size.M;
  }

  void plot(std::string title, List<DataSet> data_sets, bool persistent, bool remove, bool pause, bool lines)
  {
    // validate that gnuplot is in system path
    static bool gnuplot_on_system_path = false;
    if ((!gnuplot_on_system_path) && std::system("gnuplot --version"))
      throw std::runtime_error("gnuplot could not be found in the system path");
    gnuplot_on_system_path = true;

    // validate that x and y have the same number of elements
    for (auto& data_set : data_sets) {
      const Matrix& xdata = data_set.first;
      const Matrix& ydata = data_set.second;
      if (xdata.size().numel != ydata.size().numel) {
        std::stringstream error_message;
        error_message << "number of element mismatch (x has " << xdata.size().numel;
        error_message << " elements,  y has " << ydata.size().numel << " elements)\n";
        throw std::invalid_argument(error_message.str());
      }
    }

    // create filename
    const std::string filename = title + ".data";

    // create temporary file
    std::ofstream file(filename);

    // validate file opening
    if (!file)
      throw std::runtime_error("could not open " + filename);

    // write x and y data to file for each data set
    for (auto& data_set : data_sets) {
      const Matrix& x = data_set.first;
      const Matrix& y = data_set.second;
      for (size_t index = 0; index < x.size().numel; ++index)
        file << x[0][index] << ' ' << y[0][index] << '\n';
      // separate data sets
      file << "\n\n";
    }

    // close file
    file.close();

    // command pipeline
    std::stringstream gnuplot_pipeline;

    // launch gnuplot
    gnuplot_pipeline << "gnuplot";

    // conditionally set plot to persistent
    if (persistent)
      gnuplot_pipeline << " -persistent";

    // set plot title: -e "set title \"...\"
    gnuplot_pipeline << " -e \"set title \\\"" << title << "\\\"\"";

    // plot data: -e "plot '...'"
    gnuplot_pipeline << " -e \"plot '" << filename << (lines ? "' with lines\"" : "' \"");

    // plot remaining data sets
    for (size_t i = 1; i < data_sets.size(); ++i)
      gnuplot_pipeline << " -e \"replot '" << filename << "' index " << i << (lines ? " with lines\"" : " \"");

    // conditionally pause after plotting
    if (pause)
      gnuplot_pipeline << " -e \"pause -1 'press any key to continue...'\"";

    // execute command pipeline
    std::system(gnuplot_pipeline.str().c_str());

    // conditionally remove file after creation
    if (remove)
      std::remove(filename.c_str());
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
  /*
  using namespace Keyword;
  
  Matrix xdata = linspace(0, 5, 20, column) + Range(-0.1, 0.1);
  Matrix ydata = xdata * 2;
  auto newdata = linearize({xdata, ydata});
  
  Matrix& xnew = newdata.first;
  Matrix& ynew = newdata.second;

  Matrix speed = diff(ydata, column);
  Matrix time  = linspace(xdata(0), xdata(end), speed.size().numel);
  Matrix speednew = diff(ynew, column);
  Matrix timenew  = linspace(xnew(0), xnew(end), speednew.size().numel);
  plot("test", {{xdata, ydata},{xnew, ynew}}, true, true, false, false);
  plot("test", {{time, speed},{timenew, speednew}}, true, true, false, false);
  


  //*/

  //*
  Matrix T = {{1,  2,  3},
              {4,  5,  6},
              {7,  8,  9},
              {10, 11, 12}};
  Matrix T2({T, T});

  std::cout << T2;
  //*/
}