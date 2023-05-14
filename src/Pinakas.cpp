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
    data_(nullptr),
    numel(size_.numel)
  {
    #ifdef LOGGING
    std::clog << "Matrix created ! (empty)\n";
    #endif
  }
  
  Matrix::Matrix(const Matrix& other)
    : // member initialization list
    numel(size_.numel)
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
    data_(other.data_.release()),
    numel(size_.numel)
  {
    #ifdef LOGGING
    std::clog << "Matrix moved !\n";
    #endif
  }
    
  Matrix::Matrix(const size_t M, const size_t N)
    : // member initialization list
    numel(size_.numel)
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
    : // member initialization list
    numel(size_.numel)
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
    : // member initialization list
    numel(size_.numel)
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
    : Matrix(size.M, size.N, range)
  {}
  
  Matrix::Matrix(const List<double> list)
    : // member initialization list
    numel(size_.numel)
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
    : // member initialization list
    numel(size_.numel)
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
    : // member initialization list
    numel(size_.numel)
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
      error_message << "invalid dimensions are " << M << 'x' << N;
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
    : // member initialization list
    numel(size_.numel),
    size_{matrix.size().M, 1, matrix.size().M},
    fixed_(n),
    col_row_(false),
    matrix_(matrix)
  {}

  Slice::Slice(Matrix& matrix, const size_t m, Keyword::Row)
    : // member initialization list
    
    numel(size_.numel),
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
  Random::Random(double min, double max)
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
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] += B[0][index];
    return A;
  }
  
  Matrix operator+(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "+");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index] + B[0][index];
    return result;
  }
  
  Matrix operator+(const Matrix& A, const Random range)
  {
    Matrix result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] += A[0][index] + uniform_distribution(generator);
    return result;
  }
  
  Matrix&& operator+(Matrix&& A, const Random range)
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] += uniform_distribution(generator);
    return std::move(A);
  }
  
  Matrix&& operator+(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] += A[0][index];
    return std::move(B);
  }

  Matrix&& operator+(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] += B[0][index];
    return std::move(A);
  }
  
  Matrix&& operator+(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "+");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] += B[0][index];
    return std::move(A);
  }
  
  Matrix& operator+=(Matrix& A, const double B)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] += B;
    return A;
  }
  
  Matrix operator+(const Matrix& A, const double B)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index] + B;
    return result;
  }
  
  Matrix&& operator+(Matrix&& A, const double B)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] += B;
    return std::move(A += B);
  }
  
  Matrix operator+(const double A, const Matrix& B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A + B[0][index];
    return result;
  }
  
  Matrix&& operator+(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.numel; ++index)
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
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] *= B[0][index];
    return A;
  }
  
  Matrix operator*(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "*");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index] * B[0][index];
    return result;
  }
  
  Matrix&& operator*(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] *= A[0][index];
    return std::move(B);
  }
  
  Matrix&& operator*(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] *= B[0][index];
    return std::move(A);
  }
  
  Matrix&& operator*(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] *= B[0][index];
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix& operator*=(Matrix& A, const double B)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] *= B;
    return A;
  }
  
  Matrix operator*(const Matrix& A, const double B)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index] * B;
    return result;
  }
  
  Matrix&& operator*(Matrix&& A, const double B)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] *= B;
    return std::move(A *= B);
  }
  
  Matrix operator*(const double A, const Matrix& B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A * B[0][index];
    return result;
  }
  
  Matrix&& operator*(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] *= A;
    return std::move(B);
  }
// -------------------------------------------------------------------------------
  Matrix& operator-=(Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "-=");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] -= B[0][index];
    return A;
  }
  
  Matrix operator-(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "-");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index] - B[0][index];
    return result;
  }

  Matrix&& operator-(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] = A[0][index] - B[0][index];
    return std::move(B);
  }

  Matrix&& operator-(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] -= B[0][index];
    return std::move(A);
  }

  Matrix&& operator-(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < B.numel; ++index)
      A[0][index] -= B[0][index];
    return std::move(A);
  }

  Matrix& operator-=(Matrix& A, const double B)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] -= B;
    return A;
  }

  Matrix operator-(const Matrix& A, const double B)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index] - B;
    return result;
  }
  
  Matrix&& operator-(Matrix&& A, const double B)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] -= B;
    return std::move(A);
  }

  Matrix operator-(const double A, const Matrix& B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A - B[0][index];
    return result;
  }

  Matrix&& operator-(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] = A - B[0][index];
    return std::move(B);
  }

  Matrix operator-(const Matrix& A)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = -A[0][index];
    return result;
  }

  Matrix&& operator-(Matrix&& A)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = -A[0][index];
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix& operator/=(Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "/=");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] /= B[0][index];
    return A;
  }
  
  Matrix operator/(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "/");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index] / B[0][index];
    return result;
  }

  Matrix&& operator/(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] = A[0][index] / B[0][index];
    return std::move(B);
  }

  Matrix&& operator/(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] /= B[0][index];
    return std::move(A);
  }
  
  Matrix&& operator/(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] /= B[0][index];
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix& operator/=(Matrix& A, const double B)
  {
    const double iB = 1 / B;
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] *= iB;
    return A;
  }
  
  Matrix operator/(const Matrix& A, const double B)
  {
    const double iB = 1 / B;
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index] * iB;
    return result;
  }
  
  Matrix&& operator/(Matrix&& A, const double B)
  {
    const double iB = 1 / B;
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] *= iB;
    return std::move(A);
  }

  Matrix operator/(const double A, const Matrix& B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A / B[0][index];
    return result;
  }

  Matrix&& operator/(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] = A / B[0][index];
    return std::move(B);
  }
// -------------------------------------------------------------------------------
  Matrix& operator^=(Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "^=");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return A;
  }
  
  Matrix operator^(const Matrix& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "^");
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = std::pow(A[0][index], B[0][index]);
    return result;
  }

  Matrix&& operator^(const Matrix& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(B);
  }

  Matrix&& operator^(Matrix&& A, const Matrix& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(A);
  }

  Matrix&& operator^(Matrix&& A, Matrix&& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix& operator^=(Matrix& A, const double B)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = std::pow(A[0][index], B);
    return A;
  }

  Matrix operator^(const Matrix& A, const double B)
  {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = std::pow(A[0][index], B);
    return result;
  }
  
  Matrix&& operator^(Matrix&& A, const double B)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = std::pow(A[0][index], B);
    return std::move(A);
  }

  Matrix operator^(const double A, const Matrix& B)
  {
    Matrix result(B.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = std::pow(A, B[0][index]);
    return result;
  }

  Matrix&& operator^(const double A, Matrix&& B)
  {
    for (size_t index = 0; index < B.numel; ++index)
      B[0][index] = std::pow(A, B[0][index]);
    return std::move(B);
  }
// -------------------------------------------------------------------------------
  Matrix floor(const Matrix& A)
  {
    Matrix result(A.size(), 0);
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = std::floor(A[0][index]);
    return result;
  }

  Matrix&& floor(Matrix&& A)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = std::floor(A[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix round(const Matrix& A)
  {
    Matrix result(A.size(), 0);
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = std::round(A[0][index]);
    return result;
  }
  
  Matrix&& round(Matrix&& A)
  {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = std::round(A[0][index]);
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  Matrix ceil(Matrix& A)
  {
    Matrix result(A.size(), 0);
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = std::ceil(A[0][index]);
    return result;
  }
  
  Matrix&& ceil(Matrix&& A)
  {
    for (size_t index = 0; index < A.numel; ++index)
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

    Matrix result(A.size().M, B.size().N, 0);
    for (size_t i = 0; i < B.size().N; i++)
      for (size_t j = 0; j < A.size().M; j++)
        for (size_t k = 0; k < A.size().N; k++)
          result[j][i] += A[j][k] * B[k][i];
    return result;
  }
// -------------------------------------------------------------------------------
  Matrix transpose(const Matrix& A)
  {
    Matrix result(A.size().N, A.size().M);
    for (size_t y = 0; y < A.size().M; ++y)
      for (size_t x = 0; x < A.size().N; ++x)
        result[x][y] = A[y][x];
    return result;
  }

  Matrix reshape(const Matrix& A, const size_t M, const size_t N)
  {
    if (A.numel != M*N) {
      std::stringstream error_message;
      error_message << "reshape: can't reshape " << A.size();
      error_message << " array to " << M << 'x' << N << " array";
      throw std::invalid_argument(error_message.str());
    }
    Matrix result(M, N);
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = A[0][index];
    return result;
  }
// -------------------------------------------------------------------------------
  double min(const Matrix& matrix)
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t index = 0; index < matrix.numel; ++index)
      if (matrix[0][index] < minimum)
        minimum = matrix[0][index];
    return minimum;
  }

  double min(const Slice& column)
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t index = 0; index < column.numel; ++index)
      if (column[index] < minimum)
        minimum = column[index];
    return minimum;
  }
// -------------------------------------------------------------------------------
  double max(const Matrix& matrix)
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t index = 0; index < matrix.numel; ++index)
      if (matrix[0][index] > maximum)
        maximum = matrix[0][index];
    return maximum;
  }

  double max(const Slice& column)
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t index = 0; index < column.numel; ++index)
      if (column[index] > maximum)
        maximum = column[index];
    return maximum;
  }
// -------------------------------------------------------------------------------
  double sum(const Matrix& matrix)
  {
    double summation = 0;
    for (size_t index = 0; index < matrix.numel; ++index)
      summation += matrix[0][index];
    return summation;
  }
// -------------------------------------------------------------------------------
  double prod(const Matrix& matrix)
  {
    double product = 0;
    for (size_t index = 0; index < matrix.numel; ++index)
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
    for (i = N-1; i < N; --i) {
      // calculate appropriate Qt*b component
      substitution = 0;
      for (j = 0; j < M; ++j)
        substitution += Q[j][i] * b[j][0];

      // back substitution of previously solved x components
      for (k = N-1; k > i; --k)
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
    const size_t N = data_x.numel;
    const double step = (data_x[0][N-1] - data_x[0][0]) / (N-1);
    Matrix lin_x(data_x.size(), 0);
    Matrix lin_y(data_y.size(), 0);

    // set starting and ending value of linearized data set
    lin_x[0][0] = data_x[0][0];
    lin_y[0][0] = data_y[0][0];
    lin_x[0][N-1] = data_x[0][N-1];
    lin_y[0][N-1] = data_y[0][N-1];

    // build linearly spaced x data and its associated y value
    double x1, x2, y1, y2;
    for (size_t index = 1; index < (N-1); ++index) {
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

  Matrix linspace(const double x1, const double x2, const size_t N)
  {
    Matrix vector(1, N);
    double step = (x2 - x1) / (N-1);
    vector[0][0] = x1;
    vector[0][N-1] = x2;
    for (size_t index = 1; index < (N-1); ++index)
      vector[0][index] = vector[0][index-1] + step;
    return vector;
  }

  Matrix linspace(const double x1, const double x2, const size_t N, Keyword::Column)
  {
    Matrix vector(N, 1);
    double step = (x2 - x1) / (N-1);
    vector[0][0]     = x1;
    vector[0][N-1] = x2;
    for (size_t index = 1; index < (N-1); ++index)
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

  Matrix diff(const Matrix& A, size_t n)
  {
    if (n) {
      Matrix derivative(A.size().M, A.size().N-1, 0);
      for (size_t y = 0; y < derivative.size().M; ++y)
        for (size_t x = 0; x < derivative.size().N; ++x)
          derivative[y][x] = A[y][x + 1] - A[y][x];

      return diff(derivative, n - 1);
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

  Matrix reverse(const Matrix& A)
  {
    size_t n = A.numel;
    Matrix result(A.size());
    for (size_t index = 0; index < n; ++index)
      result[0][index] = A[0][n-1-index];
    return result;
  }

  Matrix&& reverse(Matrix&& A)
  {
    for (size_t k = 0; k < (A.numel >> 1); ++k)
      std::swap(A[0][k], A[0][A.numel-1 - k]);
    return std::move(A);
  }

  Matrix conv(const Matrix& A, const Matrix& B)
  {
    Matrix convoluted(1, A.size().N + B.size().N-1);
    for (size_t i = 0; i < A.size().N; ++i)
      for (size_t j = 0; j < B.size().N; ++j)
        convoluted[0][i + j] += A[0][i] * B[0][j];
    return convoluted;
  }
// -------------------------------------------------------------------------------
  Matrix blackman(const size_t N)
  {
    Matrix window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.42 - 0.5*std::cos(2*M_PI*k/(N-1))
                          + 0.08*std::cos(4*M_PI*k/(N-1));
    return window;
  }

  Matrix blackman(const Matrix& signal)
  {
    const size_t N = signal.numel;
    Matrix windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.42 - 0.5*std::cos(2*M_PI*k/(N-1))
                                            + 0.08*std::cos(4*M_PI*k/(N-1)));
    return windowed;
  }

  Matrix&& blackman(Matrix&& signal)
  {
    const size_t N = signal.numel;
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.42 - 0.5*std::cos(2*M_PI*k/(N-1))
                           + 0.08*std::cos(4*M_PI*k/(N-1));
    return std::move(signal);
  }
// -------------------------------------------------------------------------------
  Matrix hamming(const size_t N)
  {
    Matrix window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.54 - 0.46 * std::cos(2*M_PI*k/(N-1));
    return window;
  }

  Matrix hamming(const Matrix& signal)
  {
    const size_t N = signal.numel;
    Matrix windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.54 - 0.46 * std::cos(2*M_PI*k/(N-1)));
    return windowed;
  }

  Matrix&& hamming(Matrix&& signal)
  {
    const size_t N = signal.numel;
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.54 - 0.46 * std::cos(2*M_PI*k/(N-1));
    return std::move(signal);
  }
// -------------------------------------------------------------------------------
  Matrix hann(const size_t N)
  {
    Matrix window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.5 - 0.5*cos(2*M_PI*k/(N-1));
    return window;
  }

  Matrix hann(const Matrix& signal)
  {
    const size_t N = signal.numel;
    Matrix windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.5 - 0.5*cos(2*M_PI*k/(N-1)));
    return windowed;
  }

  Matrix&& hann(Matrix&& signal)
  {
    const size_t N = signal.numel;
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.5 - 0.5*cos(2*M_PI*k/(N-1));
    return std::move(signal);
  }
// -------------------------------------------------------------------------------
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed)
  {
    const double half_tol = tol*0.5;

    double root = seed;

    size_t iteration = 0;
    while ((tol < std::abs(function(root))) && (iteration++ < max_iteration))
      root -= tol * function(root) / (function(root + half_tol) - function(root - half_tol));

    return root;
  }

  Matrix sin(Matrix& A) {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = std::sin(A[0][index]);
    return result;
  }

  Matrix&& sin(Matrix&& A) {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = std::sin(A[0][index]);
    return std::move(A);
  }

  double sinc(const double x) {
    return x == 0 ? 1 : std::sin(M_PI*x)/(M_PI*x);
  }

  Matrix sinc(Matrix& A) {
    Matrix result(A.size());
    for (size_t index = 0; index < result.numel; ++index)
      result[0][index] = sinc(A[0][index]);
    return result;
  }

  Matrix&& sinc(Matrix&& A) {
    for (size_t index = 0; index < A.numel; ++index)
      A[0][index] = sinc(A[0][index]);
    return std::move(A);
  }

  Matrix upsample(const Matrix& data, const size_t L)
  {
    Matrix upsampled(1, L * data.numel, 0);
    for (size_t index = 0; index < data.numel; ++index)
      upsampled[0][index*L] = data[0][index];
    return upsampled;
  }

  Matrix sinc_impulse(const size_t length, const double frequency)
  {
    // validate impulse length
    if (!(length%2))
      throw std::invalid_argument("impulse length must be odd");

    // offset to the impulse center
    const signed offset = (length-1) * 0.5;

    // compute impulse
    Matrix impulse(1, length);
    for (signed k = 0; k < signed(length) ; ++k)
      impulse[0][k] = sinc((k-offset) * frequency);

    return impulse;
  }

  Matrix resample(const Matrix& data, const size_t L)
  {
    if (data.size().M != 1)
      throw std::invalid_argument("error: resample: data must be a horizontal 1-dimensional matrix");
    if (data.numel == 0)
      throw std::invalid_argument("error: resample: data must contain atleast 1 element");

    const size_t N      = data.numel;
    // offset to impulse center
	  const size_t offset = L*3.5;
    // length of impulse
	  const size_t length = 2*offset + 1;
    // indices to the first and last upsampled data elements in the symetrically extended data
    const size_t first  = L*2;
    const size_t last   = L*(N+1);
    
    // temporary variables
    size_t i, j, k;

    // symetrically extended data vector
    Matrix extended(1, (N + 4) * L, 0);
    // store and upsample left symetrical data
    extended[0][0] = 2*data[0][0] - data[0][2];
    extended[0][L] = 2*data[0][0] - data[0][1];
    // store and upsample right symetrical data
    extended[0][last + L]   = 2*data[0][N-1] - data[0][N-2];
    extended[0][last + L*2] = 2*data[0][N-1] - data[0][N-3];
    // store and upsample data
    for (k = 0; k < N; ++k)
      extended[0][L*k + first] = data[0][k];
    
    // design low-pass interpolation filter
	  const Matrix filter = blackman(sinc_impulse(length, 1.0/L));

    // interpolate upsampled data using a cropped convolution
    Matrix resampled(1, last - first + 1, 0);
    for (i = 0; i < extended.numel; ++i) {
      for (j = 0; j < filter.numel; ++j) {
        k = i + j - offset;
        // skips if the index is not within the upsampled data range
        if ((first <= k) && (k <= last))
          resampled[0][k - first] += extended[0][i] * filter[0][j];
      }
    }
       
    return resampled;
  }

  Matrix resample(const Matrix& data, const size_t L, const size_t keep, const double alpha)
  {
    if (data.size().M != 1)
      std::clog << "warning: resample: data must be a horizontal 1-dimensional matrix\n";
    if (data.numel == 0)
      throw std::invalid_argument("error: resample: data must contain atleast 1 element");
    if (L <= 1)
      throw std::invalid_argument("error: resample: L must be 2 or more");
    if (keep >= data.numel)
      throw std::invalid_argument("error: resample: keep must be less than the number of elements of data");
    if (alpha < 1)
      throw std::invalid_argument("error: resample: alpha must be bigger or equal to 1");
    const size_t N      = data.numel;
    // offset to impulse center
	  const size_t offset = L*alpha;
    // length of impulse
	  const size_t length = 2*offset + 1;
    // indices to the first and last upsampled data elements in the symetrically extended data
    const size_t first  = L*keep;
    const size_t last   = L*(keep + N-1);
    
    // temporary variables
    size_t i, j, k;

    // symetrically extended data vector
    Matrix extended(1, (N + 2*keep) * L, 0);
    // store and upsample left symetrical data
    k = 0;
    for (i = 0; i < keep; ++i) {
      extended[0][k] = 2*data[0][0] - data[0][keep - i];
      k += L;
    }
    // store and upsample data
    for (i = 0; i < N; ++i) {
      extended[0][k] = data[0][i];
      k += L;
    }
    // store and upsample right symetrical data
    for (i = 0; i < keep; ++i) {
      extended[0][k] = 2*data[0][N-1] - data[0][N-2 - i];
      k += L;
    }
    
    // design low-pass interpolation filter
	  const Matrix filter = blackman(sinc_impulse(length, 1.0/L));
    
    // interpolate upsampled data using a cropped convolution
    Matrix resampled(1, last - first + 1, 0);
    for (i = 0; i < extended.numel; ++i) {
      for (j = 0; j < filter.numel; ++j) {
        k = i + j - offset;
        // skips if the index is not within the upsampled data range
        if ((first <= k) && (k <= last))
          resampled[0][k - first] += extended[0][i] * filter[0][j];
      }
    }
       
    return resampled;
  }

  std::ostream& operator<<(std::ostream& ostream, const Matrix& A)
  {
    if (A.numel) {
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
    if (A.numel) {
      std::size_t max_len = 0;
      for (size_t y = 0; y < A.numel; ++y) {
        std::stringstream ss;
        ss.copyfmt(ostream);
        ss << A[y];
        max_len = std::max(max_len, ss.str().length());
      }
      if (A.size().M == 1) {
        ostream << std::setw(max_len) << A[0];
        for (size_t x = 1; x < A.numel; ++x)
            ostream << ' ' << std::setw(max_len) << A[x];
        ostream << '\n';
      }
      else {
        for (size_t y = 0; y < A.numel; ++y)
          ostream << std::setw(max_len) << A[y] << '\n';
      }
    }
    return ostream;
  }

  std::ostream& operator<<(std::ostream& ostream, const Size size)
  {
    return ostream << size.M << 'x' << size.N;
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
      if (xdata.numel != ydata.numel) {
        std::stringstream error_message;
        error_message << "number of element mismatch (x has " << xdata.numel;
        error_message << " elements,  y has " << ydata.numel << " elements)\n";
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
      for (size_t index = 0; index < x.numel; ++index)
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
  /*
  void plot(std::string title, List<Matrix> data, bool persistent, bool remove, bool pause, bool lines)
  {
    // validate that gnuplot is in system path
    static bool gnuplot_on_system_path = false;
    if ((!gnuplot_on_system_path) && std::system("gnuplot --version"))
      throw std::runtime_error("gnuplot could not be found in the system path");
    gnuplot_on_system_path = true;

    // create filename
    const std::string filename = title + ".data";

    // create temporary file
    std::ofstream file(filename);

    // validate file opening
    if (!file)
      throw std::runtime_error("could not open " + filename);

    // write x and y data to file for each data set
    for (auto& set : data) {
      for (size_t k = 0; k < set.numel; ++k)
        file << k << ' ' << set[0][k] << '\n';
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
    for (size_t k = 1; k < data.size(); ++k)
      gnuplot_pipeline << " -e \"replot '" << filename << "' index " << k << (lines ? " with lines\"" : " \"");

    // conditionally pause after plotting
    if (pause)
      gnuplot_pipeline << " -e \"pause -1 'press any key to continue...'\"";

    // execute command pipeline
    std::system(gnuplot_pipeline.str().c_str());

    // conditionally remove file after creation
    if (remove)
      std::remove(filename.c_str());
  }

  void plot(std::string title, Matrix data, bool persistent, bool remove, bool pause, bool lines)
  {
    plot(title, {data}, persistent, remove, pause, lines);
  }
  //*/
  double rms(const Matrix& A)
  {
    const size_t n = A.numel;
    double result = 0;
    for (size_t k = 0; k < n; ++k)
      result += A[0][k] * A[0][k];
    return result;
  }
}}
//
int main()
{
  using namespace Pinakas;
  using namespace Keyword;

  //*
  auto f = [](const Matrix& x){return (x^2) + sin(x*5)/5 - 2;};// + Random(0, 0.2);
  size_t N = 100;
  size_t L = 100;
  
  Matrix y     = f(linspace(0, 1, N));
  Matrix y_new = resample(y, L);
  plot("blackman", {{linspace(0, 1, y.numel), y}, {linspace(0, 1, y_new.numel), y_new}});
  
  //*/
  
  /*
  Matrix T = {{1,  2,  3},
              {4,  5,  6},
              {7,  8,  9},
              {10, 11, 12}};
  Matrix T2({T, T});

  std::cout << T2;
  //*/
}