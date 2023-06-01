// --inclusion guard---------------------------------------------------------------------
#define LOGGING
#include "../include/Pinakas.hpp"
#define M_PI 3.14159265358979323846
// --Pinakas library: backend forward declaration----------------------------------------
namespace Pinakas { namespace Backend
{
  bool Size::operator==(const Size B) const noexcept
  {
    return (M == B.M) && (N == B.N) && (numel == B.numel);
  }

  bool Size::operator!=(const Size B) const noexcept
  {
    return (M != B.M) ||(N != B.N) ||(numel != B.numel);
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
      for (size_t k = 0; k < size_.numel; ++k)
        data_[k] = other[0][k];
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

  template<typename T> template<typename T2>
  Matrix<T>::Matrix(const Matrix<T2>& other)
  {
#ifdef LOGGING
    std::clog << "Matrix converted !\n";
#endif

    if (size_t(this) != size_t(&other)) {
      // allocate memory
      allocate(this, other.M(), other.N());

      // store value
      for (size_t k = 0; k < size_.numel; ++k)
        data_[k] = other[0][k];
    }
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
    for (size_t k = 0; k < size_.numel; ++k)
      data_[k] = value;
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
    for (size_t k = 0; k < size_.numel; ++k)
      data_[k] = uniform_distribution(generator);
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
    size_t k = 0;
    for (const Matrix<T>& matrix : list) {
      for (size_t y = 0; y < matrix.size_.M; ++y)
        for (size_t x = 0; x < matrix.size_.N; ++x)
          data_[x + k + y * size_.N] = matrix[y][x];
      k += matrix.size_.N;
    }
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  void allocate(Matrix<T>* matrix, const size_t M, const size_t N)
  {
    // validate sizes
    if ((M == 0) ||(N == 0)) {
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
// --------------------------------------------------------------------------------------
  template<typename T>
  T* Matrix<T>::operator[](const size_t j) const noexcept
  {
    return data_.get() + j * size_.N;
  }

  template<typename T>
  T& Matrix<T>::operator()(signed int k) const
  {
    // positive and negative bound checking
    if ((k < -signed(size_.numel)) || (signed(size_.numel) <= k)) {
      std::clog << "warning: Matrix: (" << k << ") out of bound " << size_.numel << ", wrapped around to (";
      // wrap around to avoid undefined behavior
      k %= signed(size_.numel);
      std::clog << k << ")\n";
    }

    // convert negative indices
    k += (k < 0) * size_.numel;

    return data_[k];
  }

  template<typename T>
  T& Matrix<T>::operator()(signed int j, signed int i) const
  {
    // positive and negative bound checking
    if ((j < -signed(size_.M)) || (signed(size_.M) <= j)) {
      std::clog << "warning: Matrix: (" << j << ", _) out of bound " << size_.M << ", wrapped around to (";
      // convert negative indices
      j %= signed(size_.M);
      std::clog << j << ", _)\n";
    }

    // positive and negative bound checking
    if ((i < -signed(size_.N)) || (signed(size_.N) <= i)) {
      std::clog << "warning: Matrix: (_, " << i << ") out of bound " << size_.N << ", wrapped around to (_, ";
      // convert negative indices
      i %= signed(size_.N);
      std::clog << i << ")\n";
    }

    // convert negative indices
    j += (j < 0) * size_.M;
    i += (i < 0) * size_.N;

    return data_[i + j * size_.N];
  }
// --------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) &
  {
#ifdef LOGGING
    std::clog << "copy assigned\n";
#endif

    // validate both matrices are not the same
    if (this != &other) {
      // allocate memory if necessary
      if ((size_ != other.size_) ||!data_.get())
        allocate(this, other.size_.M, other.size_.N);

      // store values
      for (size_t k = 0; k < size_.numel; ++k)
        data_[k] = other[0][k];
    }
    return *this;
  }

  template<typename T>
  Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) & noexcept
  {
#ifdef LOGGING
    std::clog << "move assigned\n";
#endif

    // take over ressources from other matrix
    size_ = other.size_;
    data_.reset(other.data_.release());
    return *this;
  }

  template<typename T> template<typename T2>
  Matrix<T>& Matrix<T>::operator=(const Matrix<T2>& other) &
  {
#ifdef LOGGING
    std::clog << "conversion assigned\n";
#endif
    // allocate memory if necessary
    if ((size_ != other.size_) ||!data_.get())
      allocate(this, other.size_.M, other.size_.N);

    // store values
    for (size_t k = 0; k < size_.numel; ++k)
      data_[k] = other[0][k];

    return *this;
  }

  template<typename T>
  Matrix<T>& Matrix<T>::operator=(const T value) & noexcept
  {
#ifdef LOGGING
    std::clog << "fill assigned\n";
#endif
    // store values
    for (size_t k = 0; k < size_.numel; ++k)
      data_[k] = value;

    return *this;
  }


  template<typename T>
  Matrix<T>&& Matrix<T>::transpose(void) &&
  {
    if ((size_.M != 1) && (size_.N != 1)) {
      // transpose using new matrix
      Matrix<T> transposed(size_.N, size_.M);
      for (size_t j = 0; j < size_.M; ++j)
        for (size_t i = 0; i < size_.N; ++i)
          transposed[i][j] = data_[i + j * size_.N];

      // take over ressources transposed other matrix
      data_.reset(transposed.data_.release());
    }

    // transpose dimensions
    std::swap(size_.M, size_.N);

    return std::move(*this);
  }

  template<typename T>
  Matrix<T>& Matrix<T>::transpose(void) &
  {
    if ((size_.M != 1) && (size_.N != 1)) {
      // transpose using new matrix
      Matrix<T> transposed(size_.N, size_.M);
      for (size_t j = 0; j < size_.M; ++j)
        for (size_t i = 0; i < size_.N; ++i)
          transposed[i][j] = data_[i + j * size_.N];

      // take over ressources transposed other matrix
      data_.reset(transposed.data_.release());
    }

    // transpose dimensions
    std::swap(size_.M, size_.N);

    return *this;
  }




// --------------------------------------------------------------------------------------
  template <typename T>
  Matrix<T>::Iterator::Iterator(Matrix<T>& matrix, const size_t index) noexcept
    : // member initialization list
    matrix(matrix),
    index(index)
  {}

  template <typename T>
  bool Matrix<T>::Iterator::operator==(const Matrix<T>::Iterator& other) const noexcept
  {
    return index == other.index;
  }

  template <typename T>
  bool Matrix<T>::Iterator::operator!=(const Matrix<T>::Iterator& other) const noexcept
  {
    return this->index != other.index;
  }

  template <typename T>
  typename Matrix<T>::Iterator& Matrix<T>::Iterator::operator++(void) noexcept
  {
    ++index;
    return *this;
  }

  template <typename T>
  T& Matrix<T>::Iterator::operator*(void) const noexcept
  {
    return matrix[0][index];
  }
// --------------------------------------------------------------------------------------
  template <typename T>
  Matrix<T>::Const_Iterator::Const_Iterator(const Matrix<T>& matrix, const size_t index) noexcept
    : // member initialization list
    matrix(matrix),
    index(index)
  {}

  template <typename T>
  bool Matrix<T>::Const_Iterator::operator==(const Matrix<T>::Const_Iterator& other) const noexcept
  {
    return index == other.index;
  }

  template <typename T>
  bool Matrix<T>::Const_Iterator::operator!=(const Matrix<T>::Const_Iterator& other) const noexcept
  {
    return this->index != other.index;
  }

  template <typename T>
  typename Matrix<T>::Const_Iterator& Matrix<T>::Const_Iterator::operator++() noexcept
  {
    ++index;
    return *this;
  }

  template <typename T>
  const T& Matrix<T>::Const_Iterator::operator*(void) const noexcept
  {
    return matrix[0][index];
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  typename Matrix<T>::Iterator Matrix<T>::begin(void) noexcept
  {
    return Iterator(*this, 0);
  }

  template<typename T>
  typename Matrix<T>::Iterator Matrix<T>::end(void) noexcept
  {
    return Iterator(*this, size_.numel);
  }

  template<typename T>
  typename Matrix<T>::Const_Iterator Matrix<T>::begin(void) const noexcept
  {
    return Const_Iterator(*this, 0);
  }

  template<typename T>
  typename Matrix<T>::Const_Iterator Matrix<T>::end(void) const noexcept
  {
    return Const_Iterator(*this, size_.numel);
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Slice<T>::Slice(Matrix<T>& matrix, const size_t n, Keyword::Column) noexcept
    : // member initialization list
    size_{matrix.M(), 1, matrix.M()},
    fixed_(n),
    col_row_(false),
    matrix_(matrix)
  {}

  template<typename T>
  Slice<T>::Slice(Matrix<T>& matrix, const size_t m, Keyword::Row) noexcept
    : // member initialization list
    size_{1, matrix.N(), matrix.N()},
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
    if (index >= numel()) {
      std::stringstream error_message;
      error_message << '(' << index << ") out of bound " << numel() - 1 << " (dimensions are ";
      error_message << (col_row_ ? 1 : numel()) << 'x' << (col_row_ ? numel() : 1) << " )";
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
// --------------------------------------------------------------------------------------
  Random::Random(const double min, const double max) noexcept
    : // member initialization list
    min_(min),
    max_(max)
  {}
// --------------------------------------------------------------------------------------
  Range::Range(const size_t high) noexcept
    : // member initialization list
    start(0),
    stop(high-1),
    step_(1)
  {}

  Range::Range(const int start, const int stop) noexcept
    : // member initialization list
    start(start),
    stop(stop),
    step_(((stop-start) >= 0) ? 1 : -1)
  {}

  Range::Range(const int start, const int stop, const size_t step) noexcept
    : // member initialization list
    start(start),
    stop(stop),
    step_(((stop-start) >= 0) ? step : -signed(step))
  {}
  
  Range::Iterator::Iterator(const int value, const int step) noexcept
    : // member initialization list
    current_(value),
    step_(step)
  {}
  
  int Range::Iterator::operator*() const noexcept
  {
    return current_;
  }

  void Range::Iterator::operator++() noexcept
  {
    current_ += step_;
  }

  bool Range::Iterator::operator!=(const Iterator& other) const noexcept
  {           
    return (step_ > 0) ? (current_ <= other.current_) : (current_ >= other.current_);
  }
  
  Range::Iterator Range::begin() const noexcept
  {
    return Iterator(start, step_);
  }

  Range::Iterator Range::end() const noexcept
  {
      return Iterator(stop, step_);
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& add_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: add_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return A;
    }

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] += B[0][k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& add_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] += B;

    return A;
  }

  template<typename T>
  Matrix<T>& add_rng_inplace(Matrix<T>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] += uniform_distribution(generator);

    return A;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> add_mat(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: add_mat: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] + B[0][k];
      
    return R;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> add_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] + B;

    return R;
  }

  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> add_rng(const Matrix<T1>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);
    
    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] + uniform_distribution(generator);

    return R;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator+=(Matrix<T1>& A, const Matrix<T2>& B)
  {
    return add_mat_inplace(A, B);
  }

  template<typename T>
  Matrix<T>& operator+=(Matrix<T>& A, const Random B) noexcept
  {
    return add_rng_inplace(A, B);
  }

  template<typename T1, typename T2>
  Matrix<T1>& operator+=(Matrix<T1>& A, const T2 B) noexcept
  {
    return add_val_inplace(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator+(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return add_mat(A, B);
  }

  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator+(const Matrix<T>& A, const Random B) noexcept
  {
    return add_rng(A, B);
  }

  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator+(const Random A, const Matrix<T>& B) noexcept
  {
    return add_rng(B, A);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator+(const Matrix<T1>& A, const T2 B) noexcept
  {
    return add_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator+(const T1 A, const Matrix<T2>& B) noexcept
  {
    return add_val(B, A);
  }

  template<typename T>
  Matrix<T>& operator+(const Matrix<T>& A) noexcept
  {
    return A;
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator+(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(add_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator+(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(add_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&
  {
    return std::move(add_mat_inplace(B, A));
  }

  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&
  {
    return std::move(add_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator+(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(add_mat_inplace(A, B));
  }

  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator+(Matrix<T>&& A, const Random B) noexcept
  {
    return std::move(add_rng_inplace(A, B));
  }

  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator+(const Random A, Matrix<T>&& B) noexcept
  {
    return std::move(add_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator+(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(add_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator+(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(add_val_inplace(A, B));
  }

  template<typename T>
  Matrix<T>&& operator+(Matrix<T>&& A) noexcept
  {
    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& mul_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: mul_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return A;
    }

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] *= B[0][k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& mul_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] *= B;

    return A;
  }

  template<typename T>
  Matrix<T>& mul_rng_inplace(Matrix<T>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);
    
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] *= uniform_distribution(generator);

    return A;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> mul_mat(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: mul_mat: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] * B[0][k];

    return R;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> mul_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] * B;

    return R;
  }

  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> mul_rng(const Matrix<T1>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] * uniform_distribution(generator);

    return R;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator*=(Matrix<T1>& A, const Matrix<T2>& B)
  {
    return mul_mat_inplace(A, B);
  }

  template<typename T>
  Matrix<T>& operator*=(Matrix<T>& A, const Random B) noexcept
  {
    return mul_rng_inplace(A, B);
  }

  template<typename T1, typename T2>
  Matrix<T1>& operator*=(Matrix<T1>& A, const T2 B) noexcept
  {
    return mul_val_inplace(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator*(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return mul_mat(A, B);
  }

  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator*(const Matrix<T>& A, const Random B) noexcept
  {
    return mul_rng(A, B);
  }

  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator*(const Random A, const Matrix<T>& B) noexcept
  {
    return mul_rng(B, A);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator*(const Matrix<T1>& A, const T2 B) noexcept
  {
    return mul_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator*(const T1 A, const Matrix<T2>& B) noexcept
  {
    return mul_val(B, A);
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator*(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(mul_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator*(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(mul_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&
  {
    return std::move(mul_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&
  {
    return std::move(mul_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator*(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(mul_mat_inplace(A, B));
  }

  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator*(Matrix<T>&& A, const Random B) noexcept
  {
    return std::move(mul_rng_inplace(A, B));
  }

  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator*(const Random A, Matrix<T>&& B) noexcept
  {
    return std::move(mul_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator*(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(mul_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator*(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(mul_val_inplace(A, B));
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& sub_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: sub_ll_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return A;
    }

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] -= B[0][k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& sub_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: sub_rl_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return B;
    }

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B[0][k] = A[0][k] - B[0][k];

    return B;
  }

  template<typename T1, typename T2>
  Matrix<T1>& sub_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] -= B;

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& sub_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept
  {
    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B[0][k] = A - B[0][k];

    return B;
  }

  template<typename T>
  Matrix<T>& sub_ll_rng_inplace(Matrix<T>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] -= uniform_distribution(generator);

    return A;
  }

  template<typename T>
  Matrix<T>& sub_rl_rng_inplace(Matrix<T>& B, const Random A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B[0][k] = uniform_distribution(generator) - B[0][k];

    return B;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> sub_mat(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: sub_ll_mat: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] - B[0][k];

    return R;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> sub_ll_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] - B;

    return R;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> sub_rl_val(const Matrix<T1>& B, const T2 A) noexcept
  {
    Matrix<T3> R(B.size());

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A - B[0][k];

    return R;
  }

  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> sub_ll_rng(const Matrix<T1>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] - uniform_distribution(generator);

    return R;
  }

  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> sub_rl_rng(const Matrix<T1>& B, const Random A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    Matrix<T3> R(B.size());

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = uniform_distribution(generator) - B[0][k];

    return R;
  }

  template<typename T>
  Matrix<T>& negate_inplace(Matrix<T>& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = -A[0][k];

    return A;
  }

  template<typename T>
  Matrix<T> negate(const Matrix<T> A)
  {
    Matrix<T> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = -A[0][k];

    return R;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator-=(Matrix<T1>& A, const Matrix<T2>& B)
  {
    return sub_ll_mat_inplace(A, B);
  }
  
  template<typename T>
  Matrix<T>& operator-=(Matrix<T>& A, const Random B) noexcept
  {
    return sub_ll_rng_inplace(A, B);
  }

  template<typename T1, typename T2>
  Matrix<T1>& operator-=(Matrix<T1>& A, const T2 B) noexcept
  {
    return sub_ll_val_inplace(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator-(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return sub_ll_mat(A, B);
  }

  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator-(const Matrix<T>& A, const Random B) noexcept
  {
    return sub_ll_rng(A, B);
  }

  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator-(const Random A, const Matrix<T>& B) noexcept
  {
    return sub_rl_rng(B, A);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator-(const Matrix<T1>& A, const T2 B) noexcept
  {
    return sub_ll_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator-(const T1 A, const Matrix<T2>& B) noexcept
  {
    return sub_rl_val(B, A);
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator-(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(sub_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator-(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(sub_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&
  {
    return std::move(sub_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&
  {
    return std::move(sub_ll_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator-(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(sub_ll_mat_inplace(A, B));
  }

  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator-(Matrix<T>&& A, const Random B) noexcept
  {
    return std::move(sub_ll_rng_inplace(A, B));
  }

  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator-(const Random A, Matrix<T>&& B) noexcept
  {
    return std::move(sub_rl_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator-(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(sub_rl_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator-(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(sub_ll_val_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>& operator-(const Matrix<T>& A)
  {
    return negate(A);
  }

  template<typename T>
  Matrix<T>&& operator-(Matrix<T>&& A) noexcept
  {
    return std::move(negate_inplace(A));
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& div_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: div_ll_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return A;
    }

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] /= B[0][k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& div_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: div_ll_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return B;
    }

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B[0][k] = A[0][k] / B[0][k];

    return B;
  }

  template<typename T1, typename T2>
  Matrix<T1>& div_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const T1 iB = 1.0 / B;

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] *= iB;

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& div_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept
  {
    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B[0][k] = A / B[0][k];

    return B;
  }

  template<typename T>
  Matrix<T>& div_ll_rng_inplace(Matrix<T>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);
    
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] /= uniform_distribution(generator);

    return A;
  }

  template<typename T>
  Matrix<T>& div_rl_rng_inplace(Matrix<T>& B, const Random A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(A.min_, A.max_);

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B[0][k] = uniform_distribution(generator) / B[0][k];

    return B;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> div_ll_mat(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: div_ll_mat: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] / B[0][k];

    return R;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> div_ll_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());
    const T1 iB = 1.0 / B;

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] * iB;

    return R;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> div_rl_val(const Matrix<T1>& B, const T2 A) noexcept
  {
    Matrix<T3> R(B.size());

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A / B[0][k];

    return R;
  }

  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> div_ll_rng(const Matrix<T1>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k] / uniform_distribution(generator);

    return R;
  }

  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> div_rl_rng(const Matrix<T1>& B, const Random A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(A.min_, A.max_);

    Matrix<T3> R(B.size());

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = uniform_distribution(generator) / B[0][k];

    return R;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator/=(Matrix<T1>& A, const Matrix<T2>& B)
  {
    return div_ll_mat_inplace(A, B);
  }
  
  template<typename T>
  Matrix<T>& operator/=(Matrix<T>& A, const Random B) noexcept
  {
    return div_ll_rng_inplace(A, B);
  }

  template<typename T1, typename T2>
  Matrix<T1>& operator/=(Matrix<T1>& A, const T2 B) noexcept
  {
    return div_ll_val_inplace(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator/(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return div_ll_mat(A, B);
  }

  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator/(const Matrix<T>& A, const Random B) noexcept
  {
    return div_ll_rng(A, B);
  }

  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator/(const Random A, const Matrix<T>& B) noexcept
  {
    return div_rl_rng(B, A);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator/(const Matrix<T1>& A, const T2 B) noexcept
  {
    return div_ll_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator/(const T1 A, const Matrix<T2>& B) noexcept
  {
    return div_rl_val(B, A);
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator/(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(div_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(div_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&
  {
    return std::move(div_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&
  {
    return std::move(div_ll_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator/(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(div_ll_mat_inplace(A, B));
  }

  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator/(Matrix<T>&& A, const Random B) noexcept
  {
    return std::move(div_ll_rng_inplace(A, B));
  }

  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator/(const Random A, Matrix<T>&& B) noexcept
  {
    return std::move(div_rl_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator/(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(div_rl_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(div_ll_val_inplace(A, B));
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& pow_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: pow_ll_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return A;
    }

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::pow(A[0][k], B[0][k]);

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& pow_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: pow_ll_mat_inplace: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return B;
    }

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B[0][k] = std::pow(A[0][k], B[0][k]);

    return B;
  }

  template<typename T1, typename T2>
  Matrix<T1>& pow_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::pow(A[0][k], B);

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& pow_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept
  {
    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B[0][k] = std::pow(A, B[0][k]);

    return B;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> pow_mat(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      std::cerr << "error: pow_ll_mat: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());
	
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::pow(A[0][k], B[0][k]);

    return R;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> pow_ll_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::pow(A[0][k], B);

    return R;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> pow_rl_val(const Matrix<T1>& B, const T2 A) noexcept
  {
    Matrix<T3> R(B.size());

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::pow(A, B[0][k]);

    return R;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator^=(Matrix<T1>& A, const Matrix<T2>& B)
  {
    return pow_ll_mat_inplace(A, B);
  }

  template<typename T1, typename T2>
  Matrix<T1>& operator^=(Matrix<T1>& A, const T2 B) noexcept
  {
    return pow_ll_val_inplace(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator^(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return pow_mat(A, B);
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator^(const Matrix<T1>& A, const T2 B) noexcept
  {
    return pow_ll_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator^(const T1 A, const Matrix<T2>& B) noexcept
  {
    return pow_rl_val(B, A);
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator^(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(pow_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(pow_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&
  {
    return std::move(pow_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&
  {
    return std::move(pow_ll_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator^(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(pow_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator^(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(pow_rl_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(pow_ll_val_inplace(A, B));
  }
// --------------------------------------------------------------------------------------
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3> floor(const Matrix<T>& A)
  {
    Matrix<T> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::floor(A[0][k]);

    return R;
  }

  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3>&& floor(Matrix<T>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::floor(A[0][k]);

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3> round(const Matrix<T>& A)
  {
    Matrix<T> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::round(A[0][k]);

    return R;
  }

  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3>&& round(Matrix<T>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::round(A[0][k]);

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3> ceil(const Matrix<T>& A)
  {
    Matrix<T> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::ceil(A[0][k]);

    return R;
  }

  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3>&& ceil(Matrix<T>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::round(A[0][k]);

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<double> mul(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.N() != B.M()) {
      std::cerr << "error: mul: nonconformant arguments (A is " << A.size() << ", B is " << B.size() << ")\n";
      return Matrix<double>();
    }

    Matrix<double> result(A.M(), B.N(), 0);

    for (size_t i = 0; i < B.N(); i++)
      for (size_t j = 0; j < A.M(); j++)
        for (size_t k = 0; k < A.N(); k++)
          result[j][i] += A[j][k] * B[k][i];

    return result;
  }

  template<typename T, typename T0 = convert_to_double<T>>
  Matrix<double> div(const Matrix<T>& b, Matrix<double> A)
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
        double inorm = std::pow(sum_of_squares, -0.5);
        // normalize and store A's normalized i'th column
        for (size_t j = 0; j < M; ++j)
          Q[j][i] = A[j][i] * inorm;
      }

      // orthogonalize the remaining columns with respects to A's i'th column
      for (size_t k = i; k < N; ++k) {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        double projection = 0;
        for (size_t j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];

        // construct upper triangle matrix R using Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k >= i)
          R[i][k] = projection;

        // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k != i) // skips if k == i because the projection would be 0
          for (size_t j = 0; j < M; ++j)
            A[j][k] -= projection * Q[j][i];
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
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> transpose(const Matrix<T>& A)
  {
    Matrix<T> result(A.N(), A.M());
    for (size_t y = 0; y < A.M(); ++y)
      for (size_t x = 0; x < A.N(); ++x)
        result[x][y] = A[y][x];
    return result;
  }

  template<typename T>
  Matrix<T> reshape(const Matrix<T>& A, const size_t M, const size_t N)
  {
    if (A.numel() != M * N) {
      std::cerr << "error: reshape: can't reshape " << A.size() << " matrix to " << M << 'x' << N << " matrix";
      return Matrix<T>();
    }

    Matrix<T> R(M, N);

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][k];

    return R;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  T min(const Matrix<T>& matrix) noexcept
  {
    T minimum = std::numeric_limits<T>::max();
    for (size_t k = 0; k < matrix.numel(); ++k)
      if (matrix[0][k] < minimum)
        minimum = matrix[0][k];
        
    return minimum;
  }

  template<typename T>
  T min(const Slice<T>& slice) noexcept
  {
    T minimum = std::numeric_limits<T>::max();
    for (size_t k = 0; k < slice.numel(); ++k)
      if (slice[k] < minimum)
        minimum = slice[k];

    return minimum;
  }
  
  template<typename T>
  T max(const Matrix<T>& matrix) noexcept
  {
    T maximum = std::numeric_limits<T>::min();
    for (size_t k = 0; k < matrix.numel(); ++k)
      if (matrix[0][k] > maximum)
        maximum = matrix[0][k];

    return maximum;
  }

  template<typename T>
  T max(const Slice<T>& slice) noexcept
  {
    T maximum = std::numeric_limits<T>::min();
    for (size_t k = 0; k < slice.numel(); ++k)
      if (slice[k] > maximum)
        maximum = slice[k];

    return maximum;
  }
  
  template<typename T>
  T sum(const Matrix<T>& A) noexcept
  {
    T summation = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      summation += A[0][k];

    return summation;
  }

  template<typename T>
  double prod(const Matrix<T>& A) noexcept
  {
    double temporary = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      temporary += std::log(A[0][k]);

    return std::exp(temporary);
  }

  template<typename T>
  double avg(const Matrix<T>& A) noexcept
  {
    double average = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      average += A[0][k];

    return average/A.numel();
  }

  template<typename T>
  double rms(const Matrix<T>& A) noexcept
  {
    const size_t n = A.numel();
    double temporary = 0;
    for (size_t k = 0; k < n; ++k)
      temporary += A[0][k] * A[0][k];
    
    return std::sqrt(temporary);
  }

  template<typename T>
  double geo(const Matrix<T>& A) noexcept
  {
    double temporary = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      temporary += std::log(A[0][k]);

    return std::exp(temporary / A.numel());
  }
// --------------------------------------------------------------------------------------
  Matrix<double> orthogonalize(Matrix<double> A)
  {
    const size_t M = A.M();
    const size_t N = A.N();

    Matrix<double> Q(M, N);

    // matrix orthogonalization using the modified Gram-Schmidt process
    for (size_t i = 0; i < N; ++i) {
      // calculate the squared Euclidean norm of A's i'th column
      double sum_of_squares = 0;
      for (size_t j = 0; j < M; ++j)
        sum_of_squares += A[j][i] * A[j][i];

      // skips if the squared Euclidean norm is 0
      if (sum_of_squares != 0) {
        // calculate the inverse Euclidean norm of A's i'th column
        double inorm = std::pow(sum_of_squares, -0.5);
        // normalize and store A's normalized i'th column
        for (size_t j = 0; j < M; ++j)
          Q[j][i] = A[j][i] * inorm;
      }

      // orthogonalize the remaining columns with respects to A's i'th column
      for (size_t k = i; k < N; ++k) {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        double projection = 0;
        for (size_t j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];

        // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k != i) // skips if k == i because the projection would be 0
          for (size_t j = 0; j < M; ++j)
            A[j][k] -= projection * Q[j][i];
      }
    }

    return Q;
  }

  std::unique_ptr<Matrix<double>[]> QR(Matrix<double> A)
  {
    const size_t M = A.M();
    const size_t N = A.N();

    Matrix<double> Q(M, N);
    Matrix<double> R(N, N, 0);

    // QR decomposition using the modified Gram-Schmidt process
    for (size_t i = 0; i < N; ++i) {
      // calculate the squared Euclidean norm of A's i'th column
      double sum_of_squares = 0;
      for (size_t j = 0; j < M; ++j)
        sum_of_squares += A[j][i] * A[j][i];

      // skips if the squared Euclidean norm is 0
      if (sum_of_squares != 0) {
        // calculate the inverse Euclidean norm of A's i'th column
        double inorm = std::pow(sum_of_squares, -0.5);
        // normalize and store A's normalized i'th column
        for (size_t j = 0; j < M; ++j)
          Q[j][i] = A[j][i] * inorm;
      }

      // orthogonalize the remaining columns with respects to A's i'th column
      for (size_t k = i; k < N; ++k) {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        double projection = 0;
        for (size_t j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];

        // construct upper triangle matrix R using Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k >= i)
          R[i][k] = projection;

        // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k != i) // skips if k == i because the projection would be 0
          for (size_t j = 0; j < M; ++j)
            A[j][k] -= projection * Q[j][i];
      }
    }

    return std::unique_ptr<Matrix<double>[]>(new Matrix<double>[2]{std::move(Q), std::move(R)});
  }

  std::unique_ptr<Matrix<double>[]> linearize(const Matrix<double>& data_x, const Matrix<double>& data_y)
  {
    if (data_x.numel() != data_y.numel()) {
      std::cerr << "error: linearize: 'data_x' and 'data_y' must have the same number of elements";
      return std::unique_ptr<Matrix<double>[]>(new Matrix<double>[2]);
    }

    if ((data_x.M() != 1) ||(data_y.M() != 1))
      std::clog << "warning: linearize: data is interpreted as a horizontal 1-dimensional matrix\n";

    const size_t n = data_x.numel();

    Matrix<double> lin_x(1, n, 0);
    Matrix<double> lin_y(1, n, 0);

    // set first and last values of the linearized data set
    lin_x[0][0]   = data_x[0][0];
    lin_y[0][0]   = data_y[0][0];
    lin_x[0][n-1] = data_x[0][n-1];
    lin_y[0][n-1] = data_y[0][n-1];

    // build linearly spaced x data and its associated y value
    const double step = (data_x[0][n - 1] - data_x[0][0]) / (n - 1);
    double x1, x2, y1, y2;
    for (size_t k = 1; k < (n - 1); ++k) {
      // build linearly spaced x data
      lin_x[0][k] = lin_x[0][k - 1] + step;

      // linearly interpolate y value
      x1 = data_x[0][k];
      y1 = data_y[0][k];
      x2 = data_x[0][k+1];
      y2 = data_y[0][k+1];
      lin_y[0][k] = y1 + (lin_x[0][k] - x1) * (y2 - y1) / (x2 - x1);
    }

    return std::unique_ptr<Matrix<double>[]>(new Matrix<double>[2]{std::move(lin_x), std::move(lin_y)});;
  }

  Matrix<double> linspace(const double x1, const double x2, const size_t N)
  {
    Matrix<double> vector(1, N);

    const double step  = (x2 - x1) / (N - 1);

    const size_t n = N-1;
    vector[0][0] = x1;
    for (size_t k = 1; k < n; ++k)
      vector[0][k] = vector[0][k - 1] + step;
    vector[0][n] = x2;

    return vector;
  }

  Matrix<size_t> iota(const size_t n)
  {
    Matrix<size_t> indices(1, n);

    for (size_t k = 0; k < n; ++k)
      indices[0][k] = k;

    return indices;
  }

  template<typename T>
  Matrix<size_t> iota(const Matrix<T>& A)
  {
    const size_t n = A.numel();

    Matrix<size_t> indices(1, n);
    for (size_t k = 0; k < n; ++k)
      indices[0][k] = k;

    return indices;
  }

  Matrix<double> diff(const Matrix<double>& A, size_t n)
  {
    if (n) {
      Matrix<double> derivative(A.M(), A.N() - 1);
      for (size_t y = 0; y < derivative.M(); ++y)
        for (size_t x = 0; x < derivative.N(); ++x)
          derivative[y][x] = A[y][x + 1] - A[y][x];

      return diff(derivative, n - 1);
    }
    return A;
  }

  template<typename T>
  Matrix<T> reverse(const Matrix<T>& A)
  {
    Matrix<T> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = A[0][n-1 - k];

    return R;
  }

  template<typename T>
  Matrix<T>&& reverse(Matrix<T>&& A) noexcept
  {
    const size_t n   = A.numel();
    const size_t n_2 = n >> 1;
    for (size_t k = 0; k < n_2; ++k)
      std::swap(A[0][k], A[0][n-1 - k]);

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> conv(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    const size_t n1 = A.numel();
    const size_t n2 = B.numel();

    Matrix<T3> convoluted(1, n1 + n2 - 1, 0);
    for (size_t i = 0; i < n1; ++i)
      for (size_t j = 0; j < n2; ++j)
        convoluted[0][i + j] += A[0][i] * B[0][j];

    return convoluted;
  }

  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> corr(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    const size_t n1 = A.numel();
    const size_t n2 = B.numel();

    Matrix<T3> result(1, n1 + n2 - 1, 0);
    for (size_t i = 0; i < n1; ++i)
      for (size_t j = 0; j < n2; ++j)
        result[0][i + j] += A[0][n1-1 - i] * B[0][j];

    return result; 
  }

  template<typename T>
  Matrix<T> corr(const Matrix<T>& A)
  {
    const size_t n = A.numel();

    Matrix<T> result(1, 2*n - 1, 0);
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
    if (p >= A.numel()) {
      std::cerr << "error: lpc: p should be less than number of elements in A";
      return Matrix<double>();
    }

    if (A.M() != 1)
      std::clog << "warning: lpc: A should be a horizontal vector\n";

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
// --------------------------------------------------------------------------------------
  Matrix<double> blackman(const size_t N)
  {
    Matrix<double> window(1, N);
    for (size_t k = 0; k < N; ++k)
      window[0][k] = 0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1))
                         + 0.08 * std::cos(4 * M_PI * k / (N - 1));
    return window;
  }

  Matrix<double> blackman(const Matrix<double>& signal)
  {
    const size_t N = signal.numel();
    Matrix<double> windowed(1, N);
    for (size_t k = 0; k < N; ++k)
      windowed[0][k] = signal[0][k] * (0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1))
                                           + 0.08 * std::cos(4 * M_PI * k / (N - 1)));
    return windowed;
  }

  Matrix<double> blackman(Matrix<double>&& signal) noexcept
  {
    const size_t N = signal.numel();
    for (size_t k = 0; k < N; ++k)
      signal[0][k] *= 0.42 - 0.5 * std::cos(2 * M_PI * k / (N - 1))
                          + 0.08 * std::cos(4 * M_PI * k / (N - 1));
    return signal;
  }
// --------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
  double newton(const std::function<double(double)> function,
                const double tol,
                const size_t max_iteration,
                const double seed) noexcept
  {
    const double half_tol = tol * 0.5;

    double root = seed;

    size_t iteration = 0;
    while ((tol < std::abs(function(root))) && (iteration++ < max_iteration))
      root -= tol * function(root) / (function(root + half_tol) - function(root - half_tol));

    return root;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<double> cos(Matrix<T>& A)
  {
    Matrix<double> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::cos(A[0][k]);

    return R;
  }

  Matrix<double> cos(Matrix<double>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::cos(A[0][k]);

    return A;
  }

  template<typename T>
  Matrix<double> sin(Matrix<T>& A)
  {
    Matrix<double> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::sin(A[0][k]);

    return R;
  }

  Matrix<double> sin(Matrix<double>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::sin(A[0][k]);

    return A;
  }

  template<typename T>
  Matrix<double> sinc(Matrix<T>& A)
  {
    Matrix<double> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k) {
      double temporary = M_PI * A[0][k];
      R[0][k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return R;
  }

  Matrix<double> sinc(Matrix<double>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k) {
      double temporary = M_PI * A[0][k];
      A[0][k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return A;
  }
// --------------------------------------------------------------------------------------
  Matrix<double> upsample(const Matrix<double>& data, const size_t L)
  {
    Matrix<double> upsampled(1, L * data.numel(), 0);
    for (size_t k = 0; k < data.numel(); ++k)
      upsampled[0][k * L] = data[0][k];
    return upsampled;
  }

  Matrix<double> sinc_impulse(const size_t length, const double frequency)
  {
    // validate impulse length
    if ((length % 2) == 0) {
      std::cerr << "error: sinc_impulse: 'length' must be odd";
      return Matrix<double>();
    }

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
    if (!data.numel()) {
      std::cerr << "error: resample: 'data' must contain atleast 1 element";
      return Matrix<double>();
    }
    
    if (L <= 1) {
      std::cerr << "error: resample: 'L' must be 2 or greater";
      return Matrix<double>();
    }
    
    if (keep >= data.numel()) {
      std::cerr << "error: resample: 'keep' must be less than the number of elements in 'data'";
      return Matrix<double>();
    }
    
    if (alpha < (1.0/L)) {
      std::cerr << "error: resample: 'alpha' must be at least 1/L";
      return Matrix<double>();
    }

    if (data.M() != 1)
      std::clog << "warning: resample: 'data' is interpreted as a horizontal vector\n";
    
    const size_t n = data.numel();

    // offset to impulse center
    const size_t offset = L * alpha;
    // length of impulse
    const size_t filter_length = 2 * offset + 1;

    // indices to the first and last upsampled elements in the symetrically extended data
    const size_t first = L * keep;
    const size_t last  = L * n + first - (tail ? 1 : L);

    // symetrically extended data vector
    Matrix<double> extended(1, n + 2 * keep, 0);

    // store and upsample left symetrical data
    size_t k = 0;
    for (size_t i = 0; i < keep; ++i)
      extended[0][k++] = 2 * data[0][0] - data[0][keep - i];

    // store and upsample data
    for (size_t i = 0; i < n; ++i)
      extended[0][k++] = data[0][i];
      
    // store and upsample right symetrical data
    for (size_t i = 0; i < keep; ++i)
      extended[0][k++] = 2 * data[0][n - 1] - data[0][n - 2 - i];
      
    // design low-pass interpolation filter
    const Matrix<double> filter = blackman(sinc_impulse(filter_length, 1.0 / L));

    // interpolate upsampled data using a cropped convolution
    Matrix<double> resampled(1, last - first + 1, 0);
    for (size_t i = 0; i < extended.numel(); ++i) {
      for (size_t j = 0; j < filter_length; ++j) {
        size_t k = i*L + j - offset;
        // skips if the index is not within the upsampled data range (cropping)
        if ((first <= k) && (k <= last))
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

      for (size_t y = 0; y < A.M(); ++y) {
        for (size_t x = 0; x < A.N(); ++x) {
          std::stringstream ss;
          ss.copyfmt(ostream);
          ss << A[y][x];
          max_len = std::max(max_len, ss.str().length());
        }
      }

      for (size_t y = 0; y < A.M(); ++y) {
        for (size_t x = 0; x < A.N(); ++x)
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

  void plot(List<std::string> titles, List<DataSet> data_sets, bool persistent, bool remove, bool pause, bool lines)
  {
    // validate that gnuplot is in system path
    static bool gnuplot_on_system_path = false;
    if ((!gnuplot_on_system_path) && std::system("gnuplot --version")) {
      std::cerr << "error: plot: gnuplot could not be found in the system path";
      return;
    }
    gnuplot_on_system_path = true;

    // validate that x and y have the same number of elements
    for (auto& data_set : data_sets) {
      const Matrix<double>& xdata = data_set.first;
      const Matrix<double>& ydata = data_set.second;
      if (xdata.numel() != ydata.numel()) {
        std::cerr << "error: plot: number of element mismatch (x has " << xdata.numel();
        std::cerr << " elements,  y has " << ydata.numel() << " elements)\n";
        return;
      }
    }

    if (titles.size() != data_sets.size())
      std::clog << "warning: plot: number of titles does not equal number of data sets\n";

    // create temporary file
    std::ofstream file("gnuplot.data");

    // validate file opening
    if (!file) {
      std::cerr << "could not create/open gnuplot.data";
      return;
    }

    // write x and y data to file for each data set
    for (auto& data_set : data_sets) {
      const Matrix<double>& x = data_set.first;
      const Matrix<double>& y = data_set.second;
      for (size_t k = 0; k < x.numel(); ++k)
        file << x[0][k] << ' ' << y[0][k] << '\n';
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
    Matrix<double> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::abs(A[0][k]);

    return R;
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
    Matrix<double> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::real(A[0][k]);

    return R;
  }
  
  Matrix<double> imag(const Matrix<complex>& A)
  {
    Matrix<double> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::imag(A[0][k]);

    return R;
  }

  Matrix<complex> conj(const Matrix<complex>& A)
  {
    Matrix<complex> R(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      R[0][k] = std::conj(A[0][k]);

    return R;
  }

  Matrix<complex> conj(Matrix<complex>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A[0][k] = std::conj(A[0][k]);

    return A;
  }

  Matrix<complex>&& fft(Matrix<complex>&& signal)
  {
    const size_t N = signal.numel();

    if (N & (N - 1)) {
      std::cerr << "error: fft: the number of elements in 'signal' must be a power of 2";
      return std::move(Matrix<complex>());
    }
    
    size_t k = N; // Current stage size
    size_t n; // Size of butterfly operations
    double thetaT = M_PI / N; // Angle for twiddle factor
    complex phiT = complex(std::cos(thetaT), -std::sin(thetaT)); // Twiddle factor for the first stage

    // radix-2 decimation-in-frequency variation of the Cooley-Tukey fft algorithm
    while (k > 1) {
      n = k;
      k *= 0.5; // halve stage size
      phiT *= phiT; // Square the twiddle factor for the next stage
      complex twiddle_factor = 1; // Initialize the twiddle factor for the current stage

      // butterfly operations
      for (size_t l = 0; l < k; ++l) {
        for (size_t a = l; a < N; a += n) {
          size_t b = a + k;
          complex temporary = signal[0][a] - signal[0][b];
          signal[0][a] += signal[0][b];
          signal[0][b]  = temporary * twiddle_factor;
        }
        // Update the twiddle factor for the next butterfly operation
        twiddle_factor *= phiT;
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

    return std::move(signal);
  }

  Matrix<complex> ifft(const Matrix<complex>& spectrum)
  {
    return conj(fft(conj(spectrum))) / spectrum.numel();
  }

  Matrix<complex> ifft(Matrix<complex>&& spectrum)
  {
    return conj(fft(conj(std::move(spectrum)))) / spectrum.numel();
  }
}}
//

void sleep_for_ms(int ms)
{
  auto start = std::chrono::high_resolution_clock::now();
  while(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-start).count() < ms*1000);
}

int main()
{
  using namespace Pinakas;
  
  Matrix<double> x = {1, 2, 3};
  Matrix<double> y = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  std::cout << "mul:\n" << mul(x, y);
}