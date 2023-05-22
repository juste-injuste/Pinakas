// --inclusion guard--------------------------------------------------------------
#define LOGGING
#include "../include/Pinakas.hpp"
#define M_PI 3.14159265358979323846
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas { namespace Backend
{
  bool Size::operator==(const Size B) const noexcept
  {
    return (M == B.M) and (N == B.N) and (numel == B.numel);
  }

  bool Size::operator!=(const Size B) const noexcept
  {
    return (M != B.M) or (N != B.N) or (numel != B.numel);
  }

  template<typename T>
  Matrix<T>::~Matrix() noexcept
  {
    #ifdef LOGGING
    std::clog << "Matrix deleted !\n";
    #endif
  }

  template<typename T>
  Matrix<T>::Matrix() noexcept
    : // member initialization list
    size_{0, 0, 0},
    data_(nullptr)
  {
#ifdef LOGGING
    std::clog << "Matrix created ! (empty)\n";
#endif
  }

  template<typename T>
  Matrix<T>::Matrix(const Matrix<T>& other)
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

  template<typename T>
  Matrix<T>::Matrix(Matrix<T>&& other) noexcept
    : // member initialization list
    size_(other.size_),
    data_(other.data_.release())
  {
#ifdef LOGGING
    std::clog << "Matrix moved !\n";
#endif
  }

  template<typename T>
  Matrix<T>::Matrix(const size_t M, const size_t N)
  {
#ifdef LOGGING
    std::clog << "Matrix created !\n";
#endif

    // allocate memory
    allocate(this, M, N);
  }

  template<typename T>
  Matrix<T>::Matrix(const Size size)
    : // member initialization list
    Matrix<T>(size.M, size.N)
  {}

  template<typename T>
  Matrix<T>::Matrix(const size_t M, const size_t N, const T value)
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

  template<typename T>
  Matrix<T>::Matrix(const Size size, T value)
    : Matrix<T>(size.M, size.N, value)
  {}

  template<typename T>
  Matrix<T>::Matrix(const size_t M, const size_t N, const Random range)
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

  template<typename T>
  Matrix<T>::Matrix(const Size size, const Random range)
    : // member initialization list
    Matrix<T>(size.M, size.N, range)
  {}

  template<typename T>
  Matrix<T>::Matrix(const List<T> list)
  {
#ifdef LOGGING
    std::clog << "Matrix created !\n";
#endif

    // allocate memory
    allocate(this, 1, list.size());

    // store values
    size_t x = 0;
    for (T value : list)
      data_[x++] = value;
  }

  template<typename T>
  Matrix<T>::Matrix(const List<const List<const T>> values)
  {
#ifdef LOGGING
    std::clog << "Matrix created !\n";
#endif

    // dimension validation
    size_t temp_N = 0;
    for (const List<const T>& vector : values) {
      if (temp_N and (temp_N != vector.size())) {
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
    for (const List<const T>& vector : values) {
      size_t x = 0;
      for (T value : vector) {
        data_[x + y * size_.N] = value;
        ++x;
      }
      ++y;
    }
  }

  template<typename T>
  Matrix<T>::Matrix(const List<const Matrix<T>> list)
  {
#ifdef LOGGING
    std::clog << "Matrix created !\n";
#endif

    // dimension validation
    size_t temp_M = 0;
    size_t temp_N = 0;
    for (const Matrix<T>& matrix : list) {
      if (temp_M and (temp_M != matrix.size_.M)) {
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
    for (const Matrix<T>& matrix : list) {
      for (size_t y = 0; y < matrix.size_.M; ++y)
        for (size_t x = 0; x < matrix.size_.N; ++x)
          data_[x + index + y * size_.N] = matrix[y][x];
      index += matrix.size_.N;
    }
  }

  template<typename T>
  void allocate(Matrix<T>* matrix, const size_t M, const size_t N)
  {
    // validate sizes
    if ((M == 0) or (N == 0)) {
      std::stringstream error_message;
      error_message << "error: allocate: dimensions are " << M << 'x' << N;
      throw std::invalid_argument(error_message.str());
    }

    // allocate memory
    matrix->data_.reset((T *)new char[sizeof(T[M][N])]);

    // validate memory allocation
    if (!matrix->data_.get())
      throw std::bad_alloc();

    // save size information
    matrix->size_ = {M, N, M * N};
  }

  template<typename T>
  T* Matrix<T>::operator[](const size_t index) const noexcept
  {
    return data_.get() + index * size_.N;
  }

  template<typename T>
  T& Matrix<T>::operator()(const size_t index) const
  {
    if (index >= size_.numel) {
      std::stringstream error_message;
      error_message << '(' << index << ") out of bound " << size_.numel - 1 << " (dimensions are " << size_ << ')';
      throw std::out_of_range(error_message.str());
    }
    return data_[index];
  }

  template<typename T>
  T& Matrix<T>::operator()(Keyword::End) const noexcept
  {
    return data_[size_.numel - 1];
  }

  template<typename T>
  T& Matrix<T>::operator()(const size_t y, const size_t x) const
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

  template<typename T>
  Slice<T> Matrix<T>::operator()(Keyword::Entire, const size_t n) & noexcept
  {
    return Slice<T>(*this, n, Keyword::column);
  }

  template<typename T>
  Slice<T> Matrix<T>::operator()(const size_t m, Keyword::Entire) & noexcept
  {
    return Slice<T>(*this, m, Keyword::row);
  }

  template<typename T>
  Size Matrix<T>::size(void) const & noexcept
  {
    return size_;
  }

  template<typename T>
  size_t Matrix<T>::numel(void) const & noexcept
  {
    return size_.numel;
  }

  template<typename T>
  size_t Matrix<T>::M(void) const & noexcept
  {
    return size_.M;
  }

  template<typename T>
  size_t Matrix<T>::N(void) const & noexcept
  {
    return size_.N;
  }

  template<typename T> template<typename T2>
  Matrix<T>::operator Matrix<T2> () const
  {
    Matrix<T2> punned(size_);
    for (size_t k = 0; k < size_.numel; ++k)
      punned[0][k] = data_[k];
    return punned;
  }
  template<typename T> template<typename T2>
  Matrix<T>& Matrix<T>::operator=(const Matrix<T2>& other) &
  {
#ifdef LOGGING
    std::clog << "assigned\n";
#endif

    // validate both matrices are not the same
    if (this != &other) {
      // allocate memory if necessary
      if ((size_ != other.size_) or !data_.get())
        allocate(this, other.size_.M, other.size_.N);

      // store values
      for (size_t index = 0; index < size_.numel; ++index)
        data_[index] = other[0][index];
    }
    return *this;
  }
// -------------------------------------------------------------------------------
  template<typename T> template<typename T2>
  Matrix<T>& Matrix<T>::operator=(Matrix<T2>&& other) & noexcept
  {
#ifdef LOGGING
    std::clog << "move assigned\n";
#endif

    // take over ressources from other matrix
    size_ = other.size_;
    data_.reset(other.data_.release());
    return *this;
  }

  template<typename T>
  Matrix<T>& Matrix<T>::operator=(const T value) & noexcept
  {
    // store values
    for (size_t index = 0; index < size_.numel; ++index)
      data_[index] = value;
    return *this;
  }
// -------------------------------------------------------------------------------
  template <typename T>
  Iterator<T>::Iterator(Matrix<T>& matrix, const size_t index) noexcept
    : // member initialization list
    matrix(matrix),
    index(index)
  {}

  template <typename T>
  bool Iterator<T>::operator==(const Iterator<T>& other) const noexcept
  {
    return index == other.index;
  }

  template <typename T>
  bool Iterator<T>::operator!=(const Iterator<T>& other) const noexcept
  {
    return this->index != other.index;
  }

  template <typename T>
  Iterator<T>& Iterator<T>::operator++(void) noexcept
  {
    ++index;
    return *this;
  }

  template <typename T>
  T& Iterator<T>::operator*(void) const noexcept
  {
    return matrix[0][index];
  }
// -------------------------------------------------------------------------------
  template <typename T>
  ConstIterator<T>::ConstIterator(const T& matrix, const size_t index) noexcept
    : // member initialization list
    matrix(matrix),
    index(index)
  {}

  template <typename T>
  bool ConstIterator<T>::operator==(const ConstIterator<T>& other) const noexcept
  {
    return index == other.index;
  }

  template <typename T>
  bool ConstIterator<T>::operator!=(const ConstIterator<T>& other) const noexcept
  {
    return this->index != other.index;
  }

  template <typename T>
  ConstIterator<T>& ConstIterator<T>::operator++() noexcept
  {
    ++index;
    return *this;
  }

  template <typename T>
  const double& ConstIterator<T>::operator*(void) const noexcept
  {
    return matrix[0][index];
  }
// -------------------------------------------------------------------------------
  template<typename T>
  Iterator<Matrix<T>> Matrix<T>::begin(void) noexcept
  {
    return Iterator<Matrix<T>>(*this, 0);
  }

  template<typename T>
  Iterator<Matrix<T>> Matrix<T>::end(void) noexcept
  {
    return Iterator<Matrix<T>>(*this, size_.numel);
  }

  template<typename T>
  ConstIterator<Matrix<T>> Matrix<T>::begin(void) const noexcept
  {
    return ConstIterator<Matrix<T>>(*this, 0);
  }

  template<typename T>
  ConstIterator<Matrix<T>> Matrix<T>::end(void) const noexcept
  {
    return ConstIterator<Matrix<T>>(*this, size_.numel);
  }
// -------------------------------------------------------------------------------
  template<typename T>
  Slice<T>::Slice(Matrix<T>& matrix, const size_t n, Keyword::Column) noexcept
    : // member initialization list
    size_{matrix.size().M, 1, matrix.size().M},
    fixed_(n),
    col_row_(false),
    matrix_(matrix)
  {}

  template<typename T>
  Slice<T>::Slice(Matrix<T>& matrix, const size_t m, Keyword::Row) noexcept
    : // member initialization list
    size_{1, matrix.size().N, matrix.size().N},
    fixed_(m),
    col_row_(true),
    matrix_(matrix)
  {}

  template<typename T>
  T& Slice<T>::operator[](const size_t index) const & noexcept
  {
    return col_row_ ? matrix_[fixed_][index] : matrix_[index][fixed_];
  }

  template<typename T>
  T& Slice<T>::operator()(const size_t index) const &
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

  template<typename T>
  Size Slice<T>::size(void) const & noexcept
  {
    return size_;
  }

  template<typename T>
  size_t Slice<T>::numel(void) const & noexcept
  {
    return size_.numel;
  }
// -------------------------------------------------------------------------------
  Random::Random(double min, double max) noexcept
    : // member initialization list
    min_(min),
    max_(max)
  {}
// -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string& op)
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
  template<typename T1, typename T2>
  Matrix<T1>& add_inplace(Matrix<T1>& A, const Matrix<T2>& B) noexcept
  {
    for (size_t k = 0; k < A.numel(); ++k)
      A[0][k] += B[0][k];
    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& add_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    for (size_t k = 0; k < A.numel(); ++k)
      A[0][k] += B;
    return A;
  }

  template<typename T>
  Matrix<T>& add_inplace(Matrix<T>& A, const Random range) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t k = 0; k < A.numel(); ++k)
      A[0][k] += uniform_distribution(generator);
    return A;
  }

  template<typename T1, typename T2, typename T3 = typename std::common_type<T1, T2>::type>
  Matrix<T3> add(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    Matrix<T3> R(A.size());
    for (size_t k = 0; k < A.numel(); ++k)
      R[0][k] = A[0][k] + B[0][k];
    return R;
  }

  template<typename T1, typename T2, typename T3 = typename std::common_type<T1, T2>::type>
  Matrix<T3> add(const Matrix<T1>& A, const T2 B)
  {
    Matrix<T3> R(A.size());
    for (size_t k = 0; k < A.numel(); ++k)
      R[0][k] += A[0][k] + B;
    return R;
  }

  template<typename T1, typename T3 = typename std::common_type<T1, double>::type>
  Matrix<T3> add(const Matrix<T1>& A, const Random range)
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    Matrix<T3> R(A.size());
    for (size_t k = 0; k < A.numel(); ++k)
      R[0][k] = A[0][k] + uniform_distribution(generator);
    return R;
  }
// -------------------------------------------------------------------------------
  template<typename T1, typename T2>
  auto operator+=(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&
  {
    return add_inplace(A, B);
  }

  template<typename T>
  auto operator+=(Matrix<T>& A, const Random B) -> Matrix<T>&
  {
    return add_inplace(A, B);
  }

  template<typename T1, typename T2>
  auto operator+=(Matrix<T1>& A, const T2 B) -> Matrix<T1>&
  {
    return add_inplace(A, B);
  }
  
  template<typename T1, typename T2>
  auto operator+(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<typename std::common_type<T1, T2>::type>
  {
    return add(A, B);
  }

  template<typename T>
  auto operator+(const Matrix<T>& A, const Random B) -> Matrix<typename std::common_type<T, double>::type>
  {
    return add(A, B);
  }

  template<typename T>
  auto operator+(const Random A, const Matrix<T>& B) -> Matrix<typename std::common_type<T, double>::type>
  {
    return add(B, A);
  }
  
  template<typename T1, typename T2>
  auto operator+(const Matrix<T1>& A, const T2 B) -> Matrix<typename std::common_type<T1, T2>::type>
  {
    return add(A, B);
  }
  
  template<typename T1, typename T2>
  auto operator+(const T1 A, const Matrix<T2>& B) -> Matrix<typename std::common_type<T1, T2>::type>
  {
    return add(B, A);
  }

  template<typename T>
  Matrix<T>& operator+(const Matrix<T>& A) noexcept
  {
    return A;
  }

  template<typename T1, typename T2>
  auto operator+(const Matrix<T1>& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&
  {
    return std::move(add_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, const Matrix<T2>& B) -> Matrix<if_no_loss<T1, T2>>&&
  {
    return std::move(add_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&
  {
    return std::move(add_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&
  {
    return std::move(add_inplace(A, B));
  }
  
  template<typename T>
  auto operator+(Matrix<T>&& A, Matrix<T>&& B) -> Matrix<T>&&
  {
    return std::move(add_inplace(A, B));
  }

  template<typename T>
  auto operator+(Matrix<T>&& A, const Random B) -> Matrix<if_no_loss<T, double>>&&
  {
    return std::move(add_inplace(A, B));
  }

  template<typename T>
  auto operator+(const Random A, Matrix<T>&& B) -> Matrix<if_no_loss<T, double>>&&
  {
    return std::move(add_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator+(const T1 A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&
  {
    return std::move(add_inplace(B, A));
  }

  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, const T2 B) -> Matrix<if_no_loss<T1, T2>>&&
  {
    return std::move(add_inplace(A, B));
  }

  template<typename T>
  Matrix<T>&& operator+(Matrix<T>&& A) noexcept
  {
    return std::move(A);
  }
// -------------------------------------------------------------------------------
  template<typename T>
  Matrix<T>& operator*=(Matrix<T>& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "*=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "*");
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] * B[0][index];
    return result;
  }

  template<typename T>
  Matrix<T> operator*(const Matrix<T>& A, Matrix<T>&& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] *= A[0][index];
    return B;
  }

  template<typename T>
  Matrix<T> operator*(Matrix<T>&& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator*(Matrix<T>&& A, Matrix<T>&& B)
  {
    validate_size(A.size(), B.size(), "*");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator*(const Matrix<T>& A, const Random range)
  {
    Matrix<T> result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] * uniform_distribution(generator);
    return result;
  }

  template<typename T>
  Matrix<T> operator*(Matrix<T>&& A, const Random range) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= uniform_distribution(generator);
    return A;
  }

  template<typename T>
  Matrix<T> operator*(const Random range, const Matrix<T>& A)
  {
    Matrix<T> result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] * uniform_distribution(generator);
    return result;
  }

  template<typename T>
  Matrix<T> operator*(const Random range, Matrix<T>&& A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= uniform_distribution(generator);
    return A;
  }
  
  template<typename T>
  Matrix<T>& operator*=(Matrix<T>& A, const T B) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B;
    return A;
  }

  template<typename T>
  Matrix<T> operator*(const Matrix<T>& A, const T B)
  {
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] * B;
    return result;
  }

  template<typename T>
  Matrix<T> operator*(Matrix<T>&& A, const T B) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= B;
    return A *= B;
  }

  template<typename T>
  Matrix<T> operator*(const T A, const Matrix<T>& B)
  {
    Matrix<T> result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A * B[0][index];
    return result;
  }

  template<typename T>
  Matrix<T> operator*(const T A, Matrix<T>&& B) noexcept
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] *= A;
    return B;
  }
// -------------------------------------------------------------------------------
  template<typename T>
  Matrix<T>& operator-=(Matrix<T>& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "-=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator-(const Matrix<T>& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "-");
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] - B[0][index];
    return result;
  }

  template<typename T>
  Matrix<T> operator-(const Matrix<T>& A, Matrix<T>&& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = A[0][index] - B[0][index];
    return B;
  }

  template<typename T>
  Matrix<T> operator-(Matrix<T>&& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator-(Matrix<T>&& A, Matrix<T>&& B)
  {
    validate_size(A.size(), B.size(), "-");
    for (size_t index = 0; index < B.numel(); ++index)
      A[0][index] -= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator-(const Matrix<T>& A, const Random range)
  {
    Matrix<T> result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] - uniform_distribution(generator);
    return result;
  }

  template<typename T>
  Matrix<T> operator-(Matrix<T>&& A, const Random range) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= uniform_distribution(generator);
    return A;
  }

  template<typename T>
  Matrix<T> operator-(const Random range, const Matrix<T>& A)
  {
    Matrix<T> result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] - uniform_distribution(generator);
    return result;
  }

  template<typename T>
  Matrix<T> operator-(const Random range, Matrix<T>&& A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= uniform_distribution(generator);
    return A;
  }

  template<typename T>
  Matrix<T>& operator-=(Matrix<T>& A, const T B) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= B;
    return A;
  }

  template<typename T>
  Matrix<T> operator-(const Matrix<T>& A, const T B)
  {
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] - B;
    return result;
  }

  template<typename T>
  Matrix<T> operator-(Matrix<T>&& A, const T B) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] -= B;
    return A;
  }

  template<typename T>
  Matrix<T> operator-(const T A, const Matrix<T>& B)
  {
    Matrix<T> result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A - B[0][index];
    return result;
  }

  template<typename T>
  Matrix<T> operator-(const T A, Matrix<T>&& B) noexcept
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = A - B[0][index];
    return B;
  }

  template<typename T>
  Matrix<T> operator-(const Matrix<T>& A)
  {
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = -A[0][index];
    return result;
  }

  template<typename T>
  Matrix<T> operator-(Matrix<T>&& A) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = -A[0][index];
    return A;
  }
// -------------------------------------------------------------------------------
  template<typename T>
  Matrix<T>& operator/=(Matrix<T>& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "/=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] /= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator/(const Matrix<T>& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "/");
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] / B[0][index];
    return result;
  }

  template<typename T>
  Matrix<T> operator/(const Matrix<T>& A, Matrix<T>&& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = A[0][index] / B[0][index];
    return B;
  }

  template<typename T>
  Matrix<T> operator/(Matrix<T>&& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] /= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator/(Matrix<T>&& A, Matrix<T>&& B)
  {
    validate_size(A.size(), B.size(), "/");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] /= B[0][index];
    return A;
  }

  template<typename T>
  Matrix<T> operator/(const Matrix<T>& A, const Random range)
  {
    Matrix<T> result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] / uniform_distribution(generator);
    return result;
  }

  template<typename T>
  Matrix<T> operator/(Matrix<T>&& A, const Random range) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] /= uniform_distribution(generator);
    return A;
  }

  template<typename T>
  Matrix<T> operator/(const Random range, const Matrix<T>& A)
  {
    Matrix<T> result(A.size());
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = uniform_distribution(generator) / A[0][index];
    return result;
  }

  template<typename T>
  Matrix<T> operator/(const Random range, Matrix<T>&& A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = uniform_distribution(generator) / A[0][index];
    return A;
  }
  
  template<typename T>
  Matrix<T>& operator/=(Matrix<T>& A, const T B) noexcept
  {
    const T iB = 1 / B;
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= iB;
    return A;
  }

  template<typename T>
  Matrix<T> operator/(const Matrix<T>& A, const T B)
  {
    const T iB = T(1) / B;
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A[0][index] * iB;
    return result;
  }

  template<typename T>
  Matrix<T> operator/(Matrix<T>&& A, const T B) noexcept
  {
    const T iB = T(1) / B;
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] *= iB;
    return A;
  }

  template<typename T>
  Matrix<T> operator/(const T A, const Matrix<T>& B)
  {
    Matrix<T> result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = A / B[0][index];
    return result;
  }

  template<typename T>
  Matrix<T> operator/(const T A, Matrix<T>&& B) noexcept
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = A / B[0][index];
    return B;
  }
// -------------------------------------------------------------------------------
  template<typename T>
  Matrix<T>& operator^=(Matrix<T>& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "^=");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return A;
  }

  template<typename T>
  Matrix<T> operator^(const Matrix<T>& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "^");
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::pow(A[0][index], B[0][index]);
    return result;
  }

  template<typename T>
  Matrix<T> operator^(const Matrix<T>& A, Matrix<T>&& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = std::pow(A[0][index], B[0][index]);
    return B;
  }

  template<typename T>
  Matrix<T> operator^(Matrix<T>&& A, const Matrix<T>& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return A;
  }

  template<typename T>
  Matrix<T> operator^(Matrix<T>&& A, Matrix<T>&& B)
  {
    validate_size(A.size(), B.size(), "^");
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B[0][index]);
    return A;
  }
  
  template<typename T>
  Matrix<T>& operator^=(Matrix<T>& A, const T B) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B);
    return A;
  }

  template<typename T>
  Matrix<T> operator^(const Matrix<T>& A, const T B)
  {
    Matrix<T> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::pow(A[0][index], B);
    return result;
  }

  template<typename T>
  Matrix<T> operator^(Matrix<T>&& A, const T B) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::pow(A[0][index], B);
    return A;
  }

  template<typename T>
  Matrix<T> operator^(const T A, const Matrix<T>& B)
  {
    Matrix<T> result(B.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::pow(A, B[0][index]);
    return result;
  }

  template<typename T>
  Matrix<T> operator^(const T A, Matrix<T>&& B) noexcept
  {
    for (size_t index = 0; index < B.numel(); ++index)
      B[0][index] = std::pow(A, B[0][index]);
    return B;
  }
// -------------------------------------------------------------------------------
  Matrix<double> floor(const Matrix<double>& A)
  {
    Matrix<double> result(A.size(), 0);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::floor(A[0][index]);
    return result;
  }

  Matrix<double> floor(Matrix<double>&& A) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::floor(A[0][index]);
    return A;
  }
// -------------------------------------------------------------------------------
  Matrix<double> round(const Matrix<double>& A)
  {
    Matrix<double> result(A.size(), 0);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::round(A[0][index]);
    return result;
  }

  Matrix<double> round(Matrix<double>&& A) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::round(A[0][index]);
    return A;
  }
// -------------------------------------------------------------------------------
  Matrix<double> ceil(Matrix<double>& A)
  {
    Matrix<double> result(A.size(), 0);
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::ceil(A[0][index]);
    return result;
  }

  Matrix<double> ceil(Matrix<double>&& A) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::round(A[0][index]);
    return A;
  }
// -------------------------------------------------------------------------------
  Matrix<double> mul(const Matrix<double>& A, const Matrix<double>& B)
  {
    if (A.size().N != B.size().M) {
      std::stringstream error_message;
      error_message << "error: mul: nonconformant arguments (";
      error_message << "A is " << A.size().M << 'x' << A.size().N;
      error_message << ", B is " << B.size().M << 'x' << B.size().N << ")\n";
      throw std::invalid_argument(error_message.str());
    }

    Matrix<double> result(A.size().M, B.size().N, 0);
    for (size_t i = 0; i < B.size().N; i++)
      for (size_t j = 0; j < A.size().M; j++)
        for (size_t k = 0; k < A.size().N; k++)
          result[j][i] += A[j][k] * B[k][i];
    return result;
  }
// -------------------------------------------------------------------------------
  Matrix<double> transpose(const Matrix<double>& A)
  {
    Matrix<double> result(A.size().N, A.size().M);
    for (size_t y = 0; y < A.size().M; ++y)
      for (size_t x = 0; x < A.size().N; ++x)
        result[x][y] = A[y][x];
    return result;
  }

  Matrix<double> reshape(const Matrix<double>& A, const size_t M, const size_t N)
  {
    if (A.numel() != M * N) {
      std::stringstream error_message;
      error_message << "error: reshape: can't reshape " << A.size().M << 'x' << A.size().N;
      error_message << " array to " << M << 'x' << N << " array";
      throw std::invalid_argument(error_message.str());
    }
    Matrix<double> result(M, N);
    for (size_t k = 0; k < result.numel(); ++k)
      result[0][k] = A[0][k];
    return result;
  }
// -------------------------------------------------------------------------------
  double min(const Matrix<double>& matrix) noexcept
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t k = 0; k < matrix.numel(); ++k)
      if (matrix[0][k] < minimum)
        minimum = matrix[0][k];
    return minimum;
  }

  double min(const Slice<double>& column) noexcept
  {
    double minimum = std::numeric_limits<double>::max();
    for (size_t k = 0; k < column.numel(); ++k)
      if (column[k] < minimum)
        minimum = column[k];
    return minimum;
  }
// -------------------------------------------------------------------------------
  double max(const Matrix<double>& matrix) noexcept
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t k = 0; k < matrix.numel(); ++k)
      if (matrix[0][k] > maximum)
        maximum = matrix[0][k];
    return maximum;
  }

  double max(const Slice<double>& column) noexcept
  {
    double maximum = std::numeric_limits<double>::min();
    for (size_t k = 0; k < column.numel(); ++k)
      if (column[k] > maximum)
        maximum = column[k];
    return maximum;
  }
// -------------------------------------------------------------------------------
  double sum(const Matrix<double>& A) noexcept
  {
    double summation = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      summation += A[0][k];
    return summation;
  }

  double prod(const Matrix<double>& A) noexcept
  {
    double temporary = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      temporary += std::log(A[0][k]);
    return std::exp(temporary);
  }

  double avg(const Matrix<double>& A) noexcept
  {
    const double iN = 1/A.numel();
    double average = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      average += A[0][k] * iN;
    return average;
  }

  double rms(const Matrix<double>& A) noexcept
  {
    const size_t N = A.numel();
    double temporary = 0;
    for (size_t k = 0; k < N; ++k)
      temporary += A[0][k] * A[0][k];
    return std::sqrt(temporary);
  }

  double geo(const Matrix<double>& A) noexcept
  {
    double temporary = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      temporary += std::log(A[0][k]);
    return std::exp(temporary / A.numel());
  }
// -------------------------------------------------------------------------------
  Matrix<double> orthogonalize(Matrix<double> A)
  {
    const size_t M = A.size().M;
    const size_t N = A.size().N;

    Matrix<double> Q(M, N);

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

  std::unique_ptr<Matrix<double>[]> QR(Matrix<double> A)
  {
    const size_t M = A.size().M;
    const size_t N = A.size().N;

    std::unique_ptr<Matrix<double>[]> QR(new Matrix<double>[2]{{Matrix<double>(M, N)}, Matrix<double>(N, N, 0)});
    Matrix<double>& Q = QR[0];
    Matrix<double>& R = QR[1];

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
          for (j = 0; (k != i) and (j < M); ++j)
            A[j][k] -= projection * Q[j][i];
        if (k >= i)
          R[i][k] = projection;
      }
    }
    return QR;
  }

  Matrix<double> div(const Matrix<double>& b, Matrix<double> A)
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
    Matrix<double> Q(M, N), R(N, N), x(N, 1);

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

  std::unique_ptr<Matrix<double>[]> linearize(const Matrix<double>& data_x, const Matrix<double>& data_y)
  {
    // data is interpreted as a horizontal vector
    if ((data_x.size().M != 1) or (data_y.size().M != 1))
      std::clog << "warning: linearize: data is interpreted as a horizontal 1-dimensional matrix\n";
    // validate data set
    if (data_x.numel() != data_y.numel()) {
      std::stringstream error_message;
      error_message << "error: linearize: x and y data should have the same amount of elements";
      throw std::invalid_argument(error_message.str());
    }

    const size_t N = data_x.numel();

    std::unique_ptr<Matrix<double>[]> data_set(new Matrix<double>[2]{{Matrix<double>(1, N, 0)}, {Matrix<double>(1, N, 0)}});
    Matrix<double>& lin_x = data_set[0];
    Matrix<double>& lin_y = data_set[1];

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

  Matrix<double> linspace(const double x1, const double x2, const size_t N)
  {
    Matrix<double> vector(1, N);
    double step = (x2 - x1) / (N - 1);
    vector[0][0] = x1;
    vector[0][N - 1] = x2;
    for (size_t index = 1; index < (N - 1); ++index)
      vector[0][index] = vector[0][index - 1] + step;
    return vector;
  }

  Matrix<double> iota(const size_t N)
  {
    Matrix<double> indices(1, N);
    for (size_t index = 0; index < N; ++index)
      indices[0][index] = index;
    return indices;
  }

  Matrix<double> diff(const Matrix<double>& A, size_t n)
  {
    if (n) {
      Matrix<double> derivative(A.size().M, A.size().N - 1, 0);
      for (size_t y = 0; y < derivative.size().M; ++y)
        for (size_t x = 0; x < derivative.size().N; ++x)
          derivative[y][x] = A[y][x + 1] - A[y][x];

      return diff(derivative, n - 1);
    }
    return A;
  }

  Matrix<double> reverse(const Matrix<double>& A)
  {
    const size_t N = A.numel();
    Matrix<double> result(A.size());
    for (size_t index = 0; index < N; ++index)
      result[0][index] = A[0][N-1 - index];
    return result;
  }

  Matrix<double> reverse(Matrix<double>&& A) noexcept
  {
    const size_t N   = A.numel();
    const size_t N_2 = N >> 1;
    for (size_t k = 0; k < N_2; ++k)
      std::swap(A[0][k], A[0][N-1 - k]);
    return A;
  }
// -------------------------------------------------------------------------------
  Matrix<double> conv(const Matrix<double>& A, const Matrix<double>& B)
  {
    const size_t n1 = A.numel();
    const size_t n2 = B.numel();
    Matrix<double> convoluted(1, n1 + n2 - 1, 0);
    for (size_t i = 0; i < n1; ++i)
      for (size_t j = 0; j < n2; ++j)
        convoluted[0][i + j] += A[0][i] * B[0][j];
    return convoluted;
  }

  Matrix<double> corr(const Matrix<double>& A, const Matrix<double>& B)
  {
    const size_t n1 = A.numel();
    const size_t n2 = B.numel();
    Matrix<double> result(1, n1 + n2 - 1, 0);
    for (size_t i = 0; i < n1; ++i)
      for (size_t j = 0; j < n2; ++j)
        result[0][i + j] += A[0][n1-1 - i] * B[0][j];
    return result; 
  }

  Matrix<double> corr(const Matrix<double>& A)
  {
    const size_t n = A.numel();
    Matrix<double> result(1, 2*n - 1, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        result[0][i + j] += A[0][n-1 - i] * A[0][j];
    return result;
  }

  Matrix<double> Rxx(const Matrix<double>& A)
  {
    const size_t n = A.numel();
    Matrix<double> Rxx(1, n, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if ((i+j - n+1) < n)
          Rxx[0][i+j - n+1] += A[0][n-1 - i] * A[0][j];
    return Rxx;
  }

  Matrix<double> Rxx(const Matrix<double>& A, const size_t K)
  {
    const size_t n = A.numel();
    Matrix<double> Rxx(1, K, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if ((i+j - n+1) < K)
          Rxx[0][i+j - n+1] += A[0][n-1 - i] * A[0][j];
    return Rxx;
  }

  Matrix<double> lpc(const Matrix<double>& A, const size_t p)
  {
    if (A.size().M != 1)
      std::clog << "warning: lpc: A should be a horizontal vector\n";
    if (p >= A.numel())
      throw std::invalid_argument("error: lpc: p should be smaller than the smaller of elements in A");

    Matrix<double> rxx = Rxx(A, p+1);

    Matrix<double> autocorr(p, p);
    Matrix<double> autocorr_vec(p, 1);
    for (size_t i = 0; i < p; ++i) {
      for (size_t j = 0; j < p; ++j)
        autocorr[j][i] = rxx[0][(j > i) ? (j - i) : (i - j)];
      autocorr_vec[0][i] = rxx[0][i+1];
    }

    return div(autocorr_vec, autocorr);
  }

  Matrix<double> toeplitz(const Matrix<double>& A)
  {
    const size_t n = A.numel();
    Matrix<double> result(n, n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        result[j][i] = A[0][(j > i) ? (j - i) : (i - j)];

    return result;
  }
// -------------------------------------------------------------------------------
  Matrix<double> blackman(const size_t N)
  {
    Matrix<double> window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1)) + 0.08 * std::cos(4 * M_PI * k / (N - 1));
    return window;
  }

  Matrix<double> blackman(const Matrix<double>& signal)
  {
    const size_t N = signal.numel();
    Matrix<double> windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1)) + 0.08 * std::cos(4 * M_PI * k / (N - 1)));
    return windowed;
  }

  Matrix<double> blackman(Matrix<double>&& signal) noexcept
  {
    const size_t N = signal.numel();
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1)) + 0.08 * std::cos(4 * M_PI * k / (N - 1));
    return signal;
  }
// -------------------------------------------------------------------------------
  Matrix<double> hamming(const size_t N)
  {
    Matrix<double> window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.54 - 0.46 * std::cos(2 * M_PI * k / (N - 1));
    return window;
  }

  Matrix<double> hamming(const Matrix<double>& signal)
  {
    const size_t N = signal.numel();
    Matrix<double> windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.54 - 0.46 * std::cos(2 * M_PI * k / (N - 1)));
    return windowed;
  }

  Matrix<double> hamming(Matrix<double>&& signal) noexcept
  {
    const size_t N = signal.numel();
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.54 - 0.46 * std::cos(2 * M_PI * k / (N - 1));
    return signal;
  }
// -------------------------------------------------------------------------------
  Matrix<double> hann(const size_t N)
  {
    Matrix<double> window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.5 - 0.5 * std::cos(2 * M_PI * k / (N - 1));
    return window;
  }

  Matrix<double> hann(const Matrix<double>& signal)
  {
    const size_t N = signal.numel();
    Matrix<double> windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.5 - 0.5 * std::cos(2 * M_PI * k / (N - 1)));
    return windowed;
  }

  Matrix<double> hann(Matrix<double>&& signal) noexcept
  {
    const size_t N = signal.numel();
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.5 - 0.5 * std::cos(2 * M_PI * k / (N - 1));
    return signal;
  }
// -------------------------------------------------------------------------------
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed) noexcept
  {
    const double half_tol = tol * 0.5;

    double root = seed;

    size_t iteration = 0;
    while ((tol < std::abs(function(root))) and (iteration++ < max_iteration))
      root -= tol * function(root) / (function(root + half_tol) - function(root - half_tol));

    return root;
  }
// -------------------------------------------------------------------------------
  Matrix<double> cos(Matrix<double>& A)
  {
    Matrix<double> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::cos(A[0][index]);
    return result;
  }

  Matrix<double> cos(Matrix<double>&& A) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::cos(A[0][index]);
    return A;
  }
  Matrix<double> sin(Matrix<double>& A)
  {
    Matrix<double> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index)
      result[0][index] = std::sin(A[0][index]);
    return result;
  }

  Matrix<double> sin(Matrix<double>&& A) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index)
      A[0][index] = std::sin(A[0][index]);
    return A;
  }

  Matrix<double> sinc(Matrix<double>& A)
  {
    Matrix<double> result(A.size());
    for (size_t index = 0; index < result.numel(); ++index) {
      double temporary = M_PI * A[0][index];
      result[0][index] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }
    return result;
  }

  Matrix<double> sinc(Matrix<double>&& A) noexcept
  {
    for (size_t index = 0; index < A.numel(); ++index) {
      double temporary = M_PI * A[0][index];
      A[0][index] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }
    return A;
  }
// -------------------------------------------------------------------------------
  Matrix<double> upsample(const Matrix<double>& data, const size_t L)
  {
    Matrix<double> upsampled(1, L * data.numel(), 0);
    for (size_t index = 0; index < data.numel(); ++index)
      upsampled[0][index * L] = data[0][index];
    return upsampled;
  }

  Matrix<double> sinc_impulse(const size_t length, const double frequency)
  {
    // validate impulse length
    if ((length % 2) == 0)
      throw std::invalid_argument("error: sinc_impulse: impulse length must be odd");

    // offset to the impulse center
    const signed offset = (length - 1) * 0.5;

    // compute impulse
    Matrix<double> impulse(1, length);
    for (signed k = 0; k < signed(length); ++k) {
      double temporary = M_PI * (k - offset) * frequency;
      impulse[0][k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return impulse;
  }

  Matrix<double> resample(const Matrix<double>& data, const size_t L, const size_t keep, const double alpha, const bool tail)
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

    // indices to the first and last upsampled elements in the symetrically extended data
    const size_t first = L * keep;
    const size_t last  = L * (N - !tail) + first;

    // temporary variables
    size_t i, j, k;

    // symetrically extended data vector
    Matrix<double> extended(1, N + 2 * keep, 0);
    // store and upsample left symetrical data
    k = 0;
    for (i = 0; i < keep; ++i) {
      extended[0][k++] = 2 * data[0][0] - data[0][keep - i];
      //++k;
    }
    // store and upsample data
    for (i = 0; i < N; ++i) {
      extended[0][k++] = data[0][i];
      //++k;
    }
    // store and upsample right symetrical data
    for (i = 0; i < keep; ++i) {
      extended[0][k++] = 2 * data[0][N - 1] - data[0][N - 2 - i];
      //++k;
    }

    // design low-pass interpolation filter
    const Matrix<double> filter = blackman(sinc_impulse(length, 1.0 / L));

    // interpolate upsampled data using a cropped convolution
    Matrix<double> resampled(1, last - first + 1, 0);
    for (i = 0; i < extended.numel(); ++i) {
      for (j = 0; j < filter.numel(); ++j) {
        k = i*L + j - offset;
        // skips if the index is not within the upsampled data range
        if ((first <= k) and (k <= last))
          resampled[0][k - first] += extended[0][i] * filter[0][j];
      }
    }

    return resampled;
  }

  template<typename T>
  std::ostream& operator<<(std::ostream& ostream, const Matrix<T>& A)
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

  std::ostream& operator<<(std::ostream& ostream, const Slice<double>& A)
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

  std::ostream& operator<<(std::ostream& ostream, const Size size)
  {
    return ostream << size.M << 'x' << size.N;
  }

  void plot(std::string title, List<DataSet> data_sets, bool persistent, bool remove, bool pause, bool lines)
  {
    // validate that gnuplot is in system path
    static bool gnuplot_on_system_path = false;
    if ((!gnuplot_on_system_path) and std::system("gnuplot --version"))
      throw std::runtime_error("error: plot: gnuplot could not be found in the system path");
    gnuplot_on_system_path = true;

    // validate that x and y have the same number of elements
    for (auto& data_set : data_sets) {
      const Matrix<double>& xdata = data_set.first;
      const Matrix<double>& ydata = data_set.second;
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
    for (auto& data_set : data_sets) {
      const Matrix<double>& x = data_set.first;
      const Matrix<double>& y = data_set.second;
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
    if ((!gnuplot_on_system_path) and std::system("gnuplot --version"))
      throw std::runtime_error("error: plot: gnuplot could not be found in the system path");
    gnuplot_on_system_path = true;

    // validate that x and y have the same number of elements
    for (auto& data_set : data_sets) {
      const Matrix<double>& xdata = data_set.first;
      const Matrix<double>& ydata = data_set.second;
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
    for (auto& data_set : data_sets) {
      const Matrix<double>& x = data_set.first;
      const Matrix<double>& y = data_set.second;
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
  
  template<typename T>
  Matrix<double> abs(const Matrix<T>& A)
  {
    const size_t n = A.numel();
    Matrix<double> result(A.size());
    for (size_t k = 0; k < n; ++k)
      result[0][k] = std::abs(A[0][k]);
    return result;
  }
  
  Matrix<double> abs(Matrix<double>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::abs(A[0][k]);
    return A;
  }
  
  Matrix<double> real(const Matrix<complex>& A)
  {
    const size_t n = A.numel();
    Matrix<double> real_part(A.size());
    for (size_t k = 0; k < n; ++k)
      real_part[0][k] = std::real(A[0][k]);
    return real_part;
  }
  
  Matrix<double> imag(const Matrix<complex>& A)
  {
    const size_t n = A.numel();
    Matrix<double> imaginary_part(A.size());
    for (size_t k = 0; k < n; ++k)
      imaginary_part[0][k] = std::imag(A[0][k]);
    return imaginary_part;
  }

  Matrix<complex> conj(const Matrix<complex>& A)
  {
    const size_t n = A.numel();
    Matrix<complex> result(A.size());
    for (size_t k = 0; k < n; ++k)
      result[0][k] = std::conj(A[0][k]);
    return result;
  }

  Matrix<complex> conj(Matrix<complex>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::conj(A[0][k]);
    return A;
  }

  template<typename T>
  Matrix<complex> fft(const Matrix<T>& signal)
  {
    return fft(Matrix<complex>(signal));
  }

  Matrix<complex> fft(Matrix<complex>&& signal) noexcept
  {
    const size_t N = signal.numel(); // Size of the input array
    size_t k = N; // Current stage size
    size_t n; // Size of butterfly operations
    double thetaT = 3.14159265358979323846264338328L / N; // Angle for twiddle factor
    complex phiT = complex(std::cos(thetaT), -std::sin(thetaT)); // Twiddle factor for the first stage

    // Perform the FFT computation
    while (k > 1) {
      n = k;
      k >>= 1; // Halve the stage size
      phiT *= phiT; // Square the twiddle factor for the next stage
      complex twiddle_factor = 1.0L; // Initialize the twiddle factor for the current stage

        // Perform butterfly operations
      for (size_t l = 0; l < k; ++l) {
        for (size_t a = l; a < N; a += n) {
          size_t b = a + k; // Index of the element to combine with 'a'
          complex temporary = signal[0][a] - signal[0][b]; // Difference between 'a' and 'b'
          signal[0][a] += signal[0][b]; // Sum of 'a' and 'b'
          signal[0][b]  = temporary * twiddle_factor; // Multiply 't' by the twiddle factor
        }
        twiddle_factor *= phiT; // Update the twiddle factor for the next butterfly operation
      }
    }

    // re-order frequency bins
    const size_t bits_to_reverse = std::log2(N);
    for (size_t a = 0; a < N; ++a) {
      // b = bit reversal of a
      size_t b = a;
      b = ((b & 0xAAAAAAAA) >> 1) | ((b & 0x55555555) << 1);
      b = ((b & 0xCCCCCCCC) >> 2) | ((b & 0x33333333) << 2);
      b = ((b & 0xF0F0F0F0) >> 4) | ((b & 0x0F0F0F0F) << 4);
      b = ((b & 0xFF00FF00) >> 8) | ((b & 0x00FF00FF) << 8);
      b = ((b >> 16) | (b << 16)) >> (32 - bits_to_reverse);
      
      // swap elements
      if (b > a)
        std::swap(signal[0][a], signal[0][b]);
    }

    return signal;
  }

  Matrix<complex> ifft(const Matrix<complex>& spectrum)
  {
    return conj(fft(conj(spectrum))) / complex(spectrum.numel());
  }

  Matrix<complex> ifft(Matrix<complex>&& spectrum) noexcept
  {
    return conj(fft(conj(std::move(spectrum)))) / complex(spectrum.numel());
  }
}}
//
int main()
{
  using namespace Pinakas;
  using namespace Keyword;
  using namespace Chronometro;
  /*
  size_t N = 1 << 24;
  //size_t L = 100;
  //auto   f = [](const Matrix<double>& x) {return ((1-x)^ 2) + sin((x-0.5) * 5) / 5 - 2;};
  //auto   f = [](const Matrix<double>& x) {return sin(6.28*x*100) + sin(6.28*x*50);};

  Stopwatch sw;
  Matrix<double> x_linear = linspace(0, 1, N);
  Matrix<double> y_linear = sin(x_linear*1000) + sin(x_linear*300);//f(x_linear);

  auto y2 = (Matrix<complex>)y_linear;
  puts("------------");
  sw.start();
  y2 = fft(y2);
  sw.stop();
  puts("------------");

  auto X2 = abs(y2);
  //plot({"spectrum2", "signal"}, {{iota(X2.numel()), X2}, {iota(N), y_linear}}, true, false);
  //*/
  Matrix<int> x = iota(10) + Random(0, 1);
  Matrix<double> y = iota(10) + Random(0, 1);
  puts("----------------");
  auto t1 =  x + y;
  puts("----------------");
  auto t2 =  x + (y + 1.0);
  puts("----------------");
  auto t3 =  (x + 1) + y;
  puts("----------------");
  auto t4 =  (x + 1) + (y + 1.0);
  puts("----------------");
  
  /* 
  Matrix<double> A(1000, 1000, {0, 1});
  Matrix<double> B(1000, 1000, {0, 1});
  
  for (int k = 0; k < 10; ++k){
    Chronometro::Stopwatch sw;
    for (int i = 0; i < 10; ++i)
      A + B;
  }
  //*/


  /*
  const Matrix<double> signal = {1, 2, 3, 4};

  Matrix<complex> spectrum = fft(signal);
  std::cout << signal;
  std::cout << spectrum;
  //*/
}