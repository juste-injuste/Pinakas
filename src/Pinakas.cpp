// --inclusion guard---------------------------------------------------------------------
#include "../include/Pinakas.hpp"

#ifndef  M_PI
# define M_PI 3.14159265358979323846
#endif
// --Pinakas library: backend forward declaration----------------------------------------
namespace Pinakas
{
  namespace Backend
  {

  }

  bool Size::operator==(const Size other) const noexcept
  {
    return (numel == other.numel) && (N == other.N) && (M == other.M);
  }

  bool Size::operator!=(const Size other) const noexcept
  {
    return (numel != other.numel) || (N != other.N) || (M != other.M);
  }

  template<typename T>
  Matrix<T>::~Matrix() noexcept
  {
    delete[] data_;
    PINAKAS_LOG("deleted %ux%u", size_.M, size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix() noexcept :
    size_{0, 0, 0},
    data_(nullptr)
  {
    PINAKAS_LOG("created %ux%u", size_.M, size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const Matrix<T>& other)
  {
    if (this != &other)
    {
      // allocate memory
      allocate(other.size_.M, other.size_.N);

      // store value
      for (unsigned k = 0; k < size_.numel; ++k)
      {
        data_[k] = other[0][k];
      }
    }

    PINAKAS_LOG("copied %ux%u", other.size_.M, other.size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix(Matrix<T>&& other) noexcept :
    size_(other.size_),
    data_(other.data_)
  {
    other.size_ = Size{0, 0, 0};
    other.data_ = nullptr;

    PINAKAS_LOG("moved %ux%u", size_.M, size_.N);
  }

  template<typename T> template<typename T2>
  Matrix<T>::Matrix(const Matrix<T2>& other)
  {
    if (unsigned(this) != unsigned(&other))
    {
      // allocate memory
      allocate(other.M(), other.N());

      // store value
      for (unsigned k = 0; k < size_.numel; ++k)
      {
        data_[k] = other[0][k];
      }
    }

    PINAKAS_LOG("copy converted %ux%u", size_.M, size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const unsigned M, const unsigned N)
  {
    // allocate memory
    allocate(M, N);

    PINAKAS_LOG("created %ux%u", size_.M, size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const Size size)
    : Matrix(size.M, size.N)
  {}

  template<typename T>
  Matrix<T>::Matrix(const unsigned M, const unsigned N, const T value)
  {
    // allocate memory
    allocate(M, N);

    // store values
    for (auto& data : *this)
    {
      data = value;
    }
    // for (unsigned k = 0; k < size_.numel; ++k)
    // {
    //   data_[k] = value;
    // }

    PINAKAS_LOG("created %ux%u and filled", size_.M, size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const Size size, T value)
    : Matrix(size.M, size.N, value)
  {}

  template<typename T>
  Matrix<T>::Matrix(const unsigned M, const unsigned N, const Random range)
  {
    // allocate memory
    allocate(M, N);

    // random number generator
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(range.min_, range.max_);

    // assign random value to matrix
    for (unsigned k = 0; k < size_.numel; ++k)
      data_[k] = uniform_distribution(generator);
      
    PINAKAS_LOG("created %ux%u from range [%f, %f]", M, N, range.min_, range.max_);
  }

  template<typename T>
  Matrix<T>::Matrix(const Size size, const Random range)
    : Matrix(size.M, size.N, range)
  {}

  template<typename T>
  Matrix<T>::Matrix(const List<T> list)
  {
    // allocate memory
    allocate(1, list.size());

    // store values
    unsigned x = 0;
    for (T value : list)
      data_[x++] = value;

    PINAKAS_LOG("created %ux%u", size_.M, size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const List<const List<const T>> values)
  {
    // dimension validation
    unsigned temp_N = 0;
    for (const List<const T>& vector : values)
    {
      if (temp_N && (temp_N != vector.size()))
      {
        PINAKAS_ERROR("vertical dimensions mismatch (%u vs %u)", temp_N, unsigned(vector.size()));
        size_ = Size{0, 0, 0};
        return;
      }
      else
      {
        temp_N = vector.size();
      }
    }

    // allocate memory
    allocate(values.size(), temp_N);

    // store values
    unsigned y = 0;
    for (const auto& vector : values)
    {
      unsigned x = 0;
      for (T value : vector)
      {
        data_[x + y * size_.N] = value;
        ++x;
      }
      ++y;
    }

    PINAKAS_LOG("created %ux%u", size_.M, size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const List<const Matrix<T>> list)
  {
    // dimension validation
    unsigned M_ = 0;
    unsigned N_ = 0;
    for (const Matrix<T>& matrix : list)
    {
      if (M_ && (M_ != matrix.size_.M))
      {
        PINAKAS_ERROR("horizontal dimensions mismatch (%u vs %u)", M_, matrix.size_.M);
        size_ = Size{0, 0, 0};
        return;
      }
      else
      {
        M_ = matrix.size_.M;
      }

      N_ += matrix.size_.N;
    }

    // allocate memory
    allocate(M_, N_);

    // store values
    unsigned k = 0;
    for (const Matrix<T>& matrix : list)
    {
      for (unsigned y = 0; y < matrix.size_.M; ++y)
      {
        for (unsigned x = 0; x < matrix.size_.N; ++x)
        {
          data_[x + k + y * size_.N] = matrix[y][x];
        }
      }

      k += matrix.size_.N;
    }

    PINAKAS_LOG("created %ux%u from concatonation", size_.M, size_.N);
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  void Matrix<T>::allocate(const unsigned M, const unsigned N)
  {
    // validate sizes
    if (!(M && N))
    {
      std::stringstream error_message;
      error_message << "error: allocate: dimensions are " << M << 'x' << N;
      throw std::invalid_argument(error_message.str());
    }

    data_ = new T[M*N];

    // validate memory allocation
    if (data_ == nullptr)
      throw std::bad_alloc();

    // save size information
    size_ = {M, N, M * N};
  }
// --------------------------------------------------------------------------------------

  template<typename T>
  T* Matrix<T>::operator[](const unsigned j) noexcept
  {
    return data_ + (j * size_.N);
  }

  template<typename T>
  const T* Matrix<T>::operator[](const unsigned j) const noexcept
  {
    return data_ + (j * size_.N);
  }

  template<typename T>
  T& Matrix<T>::operator()(signed int k)
  {
    // positive and negative bound checking
    if ((k < -signed(size_.numel)) || (signed(size_.numel) <= k))
    {
      PINAKAS_WARNING("(%d) out of bound %u, wrapped around to (%d)", k, size_.numel, k %= signed(size_.numel));
    }

    // convert negative indices
    k += (k < 0) * size_.numel;

    return data_[k];
  }

  template<typename T>
  const T& Matrix<T>::operator()(signed int k) const
  {
    // positive and negative bound checking
    if ((k < -signed(size_.numel)) || (signed(size_.numel) <= k))
    {
      PINAKAS_WARNING("(%d) out of bound %u, wrapped around to (%d)", k, size_.numel, k %= signed(size_.numel));
    }

    // convert negative indices
    k += (k < 0) * size_.numel;

    return data_[k];
  }

  template<typename T>
  T& Matrix<T>::operator()(signed int j, signed int i)
  {
    // positive and negative bound checking
    if ((j < -signed(size_.M)) || (signed(size_.M) <= j))
    {
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", j, size_.M, j %= signed(size_.M));
    }

    // positive and negative bound checking
    if ((i < -signed(size_.N)) || (signed(size_.N) <= i))
    {
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", i, size_.N, i %= signed(size_.N));
    }

    // convert negative indices
    j += (j < 0) * size_.M;
    i += (i < 0) * size_.N;

    return data_[i + j * size_.N];
  }

  template<typename T>
  const T& Matrix<T>::operator()(signed int j, signed int i) const
  {
    // positive and negative bound checking
    if ((j < -signed(size_.M)) || (signed(size_.M) <= j))
    {
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", j, size_.M, j %= signed(size_.M));
    }

    // positive and negative bound checking
    if ((i < -signed(size_.N)) || (signed(size_.N) <= i))
    {
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", i, size_.N, i %= signed(size_.N));
    }

    // convert negative indices
    j += (j < 0) * size_.M;
    i += (i < 0) * size_.N;

    return data_[i + j * size_.N];
  }
  
  template<typename T>
  Slice<T> Matrix<T>::operator()(Range rows, Range cols) noexcept
  {
    if ((rows.step != 1) || (cols.step != 1))
    {
      PINAKAS_WARNING("Range step not equal to 1 note implement yet, step = 1 used instead");
      rows.step = 1;
      cols.step = 1;
    }

    // positive and negative bound checking
    if ((rows.start < -signed(size_.M)) || (signed(size_.M) <= rows.start))
    {
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", rows.start, size_.M, rows.start %= signed(size_.M));
    }

    // positive and negative bound checking
    if ((rows.stop < -signed(size_.M)) || (signed(size_.M) <= rows.stop))
    {
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", rows.stop, size_.M, rows.stop %= signed(size_.M));
    }

    // positive and negative bound checking
    if ((cols.start < -signed(size_.N)) || (signed(size_.N) <= cols.start))
    {
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", cols.start, size_.N, cols.start %= signed(size_.N));
    }

    // positive and negative bound checking
    if ((cols.stop < -signed(size_.N)) || (signed(size_.N) <= cols.stop))
    {
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", cols.stop, size_.N, cols.stop %= signed(size_.N));
    }

    // convert negative indices
    rows.start += (rows.start < 0) * size_.M;
    rows.stop  += (rows.stop < 0)  * size_.M;
    cols.start += (cols.start < 0) * size_.N;
    cols.stop  += (cols.stop < 0)  * size_.N;

    return Slice<T>(data_, size_, rows, cols);
  }
  
  template<typename T>
  Slice<T> Matrix<T>::operator()(Range rows, signed int col) noexcept
  {
    return operator()(rows, {col, col});
  }
  
  template<typename T>
  Slice<T> Matrix<T>::operator()(signed int row, Range cols) noexcept
  {
    return operator()({row, row}, cols);
  }

  template<typename T>
  Slice<const T> Matrix<T>::operator()(Range rows, Range cols) const noexcept
  {
    // positive and negative bound checking
    if ((rows.start < -signed(size_.M)) || (signed(size_.M) <= rows.start))
    {
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", rows.start, size_.M, rows.start %= signed(size_.M));
    }

    // positive and negative bound checking
    if ((rows.stop < -signed(size_.M)) || (signed(size_.M) <= rows.stop))
    {
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", rows.stop, size_.M, rows.stop %= signed(size_.M));
    }

    // positive and negative bound checking
    if ((cols.start < -signed(size_.N)) || (signed(size_.N) <= cols.start))
    {
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", cols.start, size_.N, cols.start %= signed(size_.N));
    }

    // positive and negative bound checking
    if ((cols.stop < -signed(size_.N)) || (signed(size_.N) <= cols.stop))
    {
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", cols.stop, size_.N, cols.stop %= signed(size_.N));
    }

    // convert negative indices
    rows.start += (rows.start < 0) * size_.M;
    rows.stop  += (rows.stop < 0)  * size_.M;
    cols.start += (cols.start < 0) * size_.N;
    cols.stop  += (cols.stop < 0)  * size_.N;

    return Slice<const T>(data_, size_, rows, cols);
  }
  
  template<typename T>
  Slice<const T> Matrix<T>::operator()(Range rows, signed int col) const noexcept
  {
    return operator()(rows, {col, col});
  }
  
  template<typename T>
  Slice<const T> Matrix<T>::operator()(signed int row, Range cols) const noexcept
  {
    return operator()({row, row}, cols);
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Size Matrix<T>::size(void) const & noexcept
  {
    return size_;
  }

  template<typename T>
  unsigned Matrix<T>::numel(void) const & noexcept
  {
    return size_.numel;
  }

  template<typename T>
  unsigned Matrix<T>::M(void) const & noexcept
  {
    return size_.M;
  }

  template<typename T>
  unsigned Matrix<T>::N(void) const & noexcept
  {
    return size_.N;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) &
  {
    PINAKAS_LOG("copy assigned");

    // validate both matrices are not the same
    if (this != &other) {
      // allocate memory if necessary
      if ((size_ != other.size_) ||!data_)
        allocate(other.size_.M, other.size_.N);

      // store values
      for (unsigned k = 0; k < size_.numel; ++k)
        data_[k] = other[0][k];
    }
    return *this;
  }

  template<typename T>
  Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) & noexcept
  {
    if (other.data_ != nullptr)
    {
      // take over ressources from other matrix
      size_ = other.size_;
      data_ = other.data_;

      other.size_ = Size{0, 0, 0};
      other.data_ = nullptr;
      
      PINAKAS_LOG("move assigned %ux%u", size_.M, size_.N);
    }
    else PINAKAS_LOG("nothing to move");

    return *this;
  }

  template<typename T> template<typename T2>
  Matrix<T>& Matrix<T>::operator=(const Matrix<T2>& other) &
  {
    PINAKAS_LOG("conversion assigned");

    // allocate memory if necessary
    if ((size_ != other.size_) ||!data_)
    {
      allocate(other.size_.M, other.size_.N);
    }

    // store values
    for (unsigned k = 0; k < size_.numel; ++k)
      data_[k] = other[0][k];

    return *this;
  }

  template<typename T>
  Matrix<T>& Matrix<T>::operator=(const T value) & noexcept
  {
    PINAKAS_LOG("filled");

    // store values
    for (unsigned k = 0; k < size_.numel; ++k)
      data_[k] = value;

    return *this;
  }

  template<typename T>
  Matrix<T>::operator unsigned () const noexcept
  {
    return size_.numel;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  typename Matrix<T>::iterator Matrix<T>::begin(void) noexcept
  {
    return data_;
  }

  template<typename T>
  typename Matrix<T>::iterator Matrix<T>::end(void) noexcept
  {
    return data_ + size_.numel;
  }

  template<typename T>
  typename Matrix<T>::const_iterator Matrix<T>::begin(void) const noexcept
  {
    return data_;
  }

  template<typename T>
  typename Matrix<T>::const_iterator Matrix<T>::end(void) const noexcept
  {
    return data_ + size_.numel;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Slice<T>::Slice(T* matrix_data, const Size matrix_size, const Range rows, const Range cols) noexcept :
    matrix_data_(matrix_data),
    matrix_M_(matrix_size.M),
    offset_(rows.start * matrix_size.N + cols.start),
    size_{unsigned(rows.stop - rows.start + 1)
        , unsigned(cols.stop - cols.start + 1)
        , unsigned((rows.stop - rows.start + 1) * (cols.stop - cols.start + 1))}
  {}

  template<typename T>
  Slice<T>::Slice(const Slice<T>& other) noexcept :
    matrix_data_(other.matrix_data_),
    matrix_M_(other.matrix_M_),
    offset_(other.offset_),
    size_(other.size_)
  {}

  template<typename T>
  Slice<T>& Slice<T>::operator=(const Slice<T>& other)
  {
    if (size_ != other.size())
    {
      PINAKAS_ERROR("incompatible sizes (%ux%u vs %ux%u)", size_.M, size_.N, other.size_.M, other.size_.N);
      return *this;
    }
    
    for (unsigned j = 0; j < size_.M; ++j)
    {
      for (unsigned i = 0; i < size_.N; ++i)
      {
        matrix_data_[i + j*matrix_M_ + offset_] = other[j][i];
      }
    }

    return *this;
  }

  template<typename T>
  Slice<T>& Slice<T>::operator=(const Matrix<T>& other)
  {
    if (size_ != other.size())
    {
      PINAKAS_ERROR("incompatible sizes (%ux%u vs %ux%u)", size_.M, size_.N, other.size_.M, other.size_.N);
      return *this;
    }
    
    for (unsigned j = 0; j < size_.M; ++j)
      for (unsigned i = 0; i < size_.N; ++i)
        matrix_data_[i + j*matrix_M_ + offset_] = other[j][i];

    return *this;
  }

  template<typename T>
  Size Slice<T>::size(void) const & noexcept
  {
    return size_;
  }

  template<typename T>
  unsigned Slice<T>::M(void) const & noexcept
  {
    return size_.M;
  }

  template<typename T>
  unsigned Slice<T>::N(void) const & noexcept
  {
    return size_.N;
  }

  template<typename T>
  unsigned Slice<T>::numel(void) const & noexcept
  {
    return size_.numel;
  }

  template<typename T>
  T* Slice<T>::operator[](const unsigned j) noexcept
  {
    return matrix_data_ + (j*matrix_M_ + offset_);
  }

  template<typename T>
  const T* Slice<T>::operator[](const unsigned j) const noexcept
  {
    return matrix_data_ + (j*matrix_M_ + offset_);
  }

  template<typename T>
  T& Slice<T>::operator()(signed int k) noexcept
  {
    // positive and negative bound checking
    if ((k < -signed(size_.numel)) || (signed(size_.numel) <= k))
    {
      PINAKAS_WARNING("(%d) out of bound %u, wrapped around to (%d)", k, size_.numel, k %= signed(size_.numel));
    }

    // convert negative indices
    k += (k < 0) * size_.numel;

    // compute 2D indices
    const unsigned i = k % size_.N;
    const unsigned j = k / size_.N;

    return matrix_data_[i + j*matrix_M_ + offset_];
  }

  template<typename T>
  const T& Slice<T>::operator()(signed int k) const noexcept
  {
    // positive and negative bound checking
    if ((k < -signed(size_.numel)) || (signed(size_.numel) <= k))
    {
      PINAKAS_WARNING("(%d) out of bound %u, wrapped around to (%d)", k, size_.numel, k %= signed(size_.numel));
    }

    // convert negative indices
    k += (k < 0) * size_.numel;

    // compute 2D indices
    const unsigned i = k % size_.N;
    const unsigned j = k / size_.N;

    return matrix_data_[i + j*matrix_M_ + offset_];
  }

  template<typename T>
  T& Slice<T>::operator()(signed int j, signed int i) noexcept
  {
    // positive and negative bound checking
    if ((j < -signed(size_.M)) || (signed(size_.M) <= j))
    {
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", j, size_.M, j %= signed(size_.M));
    }

    // positive and negative bound checking
    if ((i < -signed(size_.N)) || (signed(size_.N) <= i))
    {
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", i, size_.N, i %= signed(size_.N));
    }

    // convert negative indices
    j += (j < 0) * size_.M;
    i += (i < 0) * size_.N;

    return matrix_data_[i + j*matrix_M_ + offset_];
  }

  template<typename T>
  const T& Slice<T>::operator()(signed int j, signed int i) const noexcept
  {
    // positive and negative bound checking
    if ((j < -signed(size_.M)) || (signed(size_.M) <= j))
    {
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", j, size_.M, j %= signed(size_.M));
    }

    // positive and negative bound checking
    if ((i < -signed(size_.N)) || (signed(size_.N) <= i))
    {
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", i, size_.N, i %= signed(size_.N));
    }

    // convert negative indices
    j += (j < 0) * size_.M;
    i += (i < 0) * size_.N;

    return matrix_data_[i + j*matrix_M_ + offset_];
  }
  
  template<typename T> template<typename T0>
  Slice<T>::Iterator<T0>::Iterator(Slice<T>& slice, const int value) noexcept :
    slice_(slice),
    current_(value)
  {}
  
  template<typename T> template<typename T0>
  T0& Slice<T>::Iterator<T0>::operator*() const noexcept
  {
    const unsigned i = current_ % slice_.size_.N;
    const unsigned j = current_ / slice_.size_.N;

    return slice_.matrix_data_[i + j*slice_.matrix_M_ + slice_.offset_];
  }

  template<typename T> template<typename T0>
  void Slice<T>::Iterator<T0>::operator++() noexcept
  {
    ++current_;
  }

  template<typename T> template<typename T0>
  bool Slice<T>::Iterator<T0>::operator!=(const Iterator& other) const noexcept
  {           
    return current_ != other.current_;
  }
  
  template<typename T>
  typename Slice<T>::template Iterator<T> Slice<T>::begin() noexcept
  {
    return Iterator<T>(*this, 0);
  }

  template<typename T>
  typename Slice<T>::template Iterator<T> Slice<T>::end() noexcept
  {
      return Iterator<T>(*this, size_.numel);
  }
  
  template<typename T>
  typename Slice<T>::template Iterator<const T> Slice<T>::begin() const noexcept
  {
    return Iterator<const T>(*this, 0);
  }

  template<typename T>
  typename Slice<T>::template Iterator<const T> Slice<T>::end() const noexcept
  {
      return Iterator<const T>(*this, size_.numel);
  }
// --------------------------------------------------------------------------------------
  Random::Random(const double min, const double max) noexcept :
    min_(min),
    max_(max)
  {}
// --------------------------------------------------------------------------------------
  Range::Range(const unsigned high) noexcept :
    start(0),
    stop(high-1),
    step(1)
  {}

  Range::Range(const int start, const int stop) noexcept :
    start(start),
    stop(stop),
    step(((stop-start) >= 0) ? 1 : -1)
  {}

  Range::Range(const int start, const int stop, const unsigned step) noexcept :
    start(start),
    stop(stop),
    step(((stop-start) >= 0) ? step : -signed(step))
  {}
  
  Range::Iterator::Iterator(const int value, const int step) noexcept :
    current(value),
    step(step)
  {}
  
  int Range::Iterator::operator*() const noexcept
  {
    return current;
  }

  void Range::Iterator::operator++() noexcept
  {
    current += step;
  }

  bool Range::Iterator::operator!=(const Iterator& other) const noexcept
  {           
    return (step > 0) ? (current <= other.current) : (current >= other.current);
  }
  
  Range::Iterator Range::begin() const noexcept
  {
    return Iterator(start, step);
  }

  Range::Iterator Range::end() const noexcept
  {
      return Iterator(stop, step);
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& add_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size())
    {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return A;
    }

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] += B[0][k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& add_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] += B;

    return A;
  }

  template<typename T>
  Matrix<T>& add_rng_inplace(Matrix<T>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] += uniform_distribution(generator);

    return A;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> add_mat_sequ(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    Matrix<T3> R;
    if (A.size() != B.size())
    {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return R;
    }

    R = Matrix<T3>(A.size());

    const T1* a = A.data();
    const T2* b = B.data();
    T3* r = R.data();

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      r[k] = a[k] + b[k];
    }
    
    return R;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> add_val_sequ(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());

    const T1* a = A.data();
    T3* r = R.data();

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      r[k] = a[k] + B;

    return R;
  }

  template<typename T1, typename T3>
  Matrix<T3> add_rng(const Matrix<T1>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);
    
    Matrix<T3> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
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
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator+(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return add_mat_sequ(A, B);
  }
  
  template<typename T>
  Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B)
  {
    #ifdef PARALLILOS_USE_PARALLELISM
      // return add_mat_simd(A, B);
    #else
      return add_mat_sequ(A, B);
    #endif
  }

  template<typename T, typename T3>
  Matrix<T3> operator+(const Matrix<T>& A, const Random B) noexcept
  {
    return add_rng(A, B);
  }

  template<typename T, typename T3>
  Matrix<T3> operator+(const Random A, const Matrix<T>& B) noexcept
  {
    return add_rng(B, A);
  }

  template<typename T>
  Matrix<T> operator+(const Matrix<T>& A, const T B) noexcept
  {
    #ifdef PARALLILOS_USE_PARALLELISM
      // return add_val_simd(A, B);
    #else
      return add_val_sequ(A, B);
    #endif
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator+(const Matrix<T1>& A, const T2 B) noexcept
  {
    return add_val(A, B);
  }
  
  template<typename T>
  Matrix<T> operator+(const T A, const Matrix<T>& B) noexcept
  {
    #ifdef PARALLILOS_USE_PARALLELISM
      // return add_val_simd(B, A);
    #else
      return add_val_sequ(B, A);
    #endif
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator+(const T1 A, const Matrix<T2>& B) noexcept
  {
    return add_val(B, A);
  }

  template<typename T>
  Matrix<T>& operator+(const Matrix<T>& A) noexcept
  {
    return A;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator+(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(add_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator+(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(add_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
  {
    return std::move(add_mat_inplace(B, A));
  }

  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
  {
    return std::move(add_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator+(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(add_mat_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator+(Matrix<T>&& A, const Random B) noexcept
  {
    return std::move(add_rng_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator+(const Random A, Matrix<T>&& B) noexcept
  {
    return std::move(add_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator+(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(add_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3>
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
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return A;
    }

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] *= B[0][k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& mul_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] *= B;

    return A;
  }

  template<typename T>
  Matrix<T>& mul_rng_inplace(Matrix<T>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);
    
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] *= uniform_distribution(generator);

    return A;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> mul_mat_sequ(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());

    const T1* a = A.data();
    const T1* b = B.data();
    T1* r = R.data();

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      r[k] = a[k] * b[k];
    
    return R;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> mul_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = A[0][k] * B;

    return R;
  }

  template<typename T1, typename T3>
  Matrix<T3> mul_rng(const Matrix<T1>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    Matrix<T3> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
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
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator*(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return mul_mat_sequ(A, B);
  }
  
  template<typename T>
  Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B)
  {
    #ifdef PARALLILOS_USE_PARALLELISM
      // return mul_mat_simd(A, B);
    #else
      return mul_mat_sequ(A, B);
    #endif
  }

  template<typename T, typename T3>
  Matrix<T3> operator*(const Matrix<T>& A, const Random B) noexcept
  {
    return mul_rng(A, B);
  }

  template<typename T, typename T3>
  Matrix<T3> operator*(const Random A, const Matrix<T>& B) noexcept
  {
    return mul_rng(B, A);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator*(const Matrix<T1>& A, const T2 B) noexcept
  {
    return mul_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator*(const T1 A, const Matrix<T2>& B) noexcept
  {
    return mul_val(B, A);
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator*(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(mul_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator*(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(mul_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
  {
    return std::move(mul_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
  {
    return std::move(mul_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator*(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(mul_mat_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator*(Matrix<T>&& A, const Random B) noexcept
  {
    return std::move(mul_rng_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator*(const Random A, Matrix<T>&& B) noexcept
  {
    return std::move(mul_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator*(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(mul_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator*(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(mul_val_inplace(A, B));
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& sub_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return A;
    }

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] -= B[0][k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& sub_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return B;
    }

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      B[0][k] = A[0][k] - B[0][k];

    return B;
  }

  template<typename T1, typename T2>
  Matrix<T1>& sub_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] -= B;

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& sub_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept
  {
    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      B[0][k] = A - B[0][k];

    return B;
  }

  template<typename T>
  Matrix<T>& sub_ll_rng_inplace(Matrix<T>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] -= uniform_distribution(generator);

    return A;
  }

  template<typename T>
  Matrix<T>& sub_rl_rng_inplace(Matrix<T>& B, const Random A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      B[0][k] = uniform_distribution(generator) - B[0][k];

    return B;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> sub_mat_sequ(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());

    const T1* a = A.data();
    const T1* b = B.data();
    T1* r = R.data();

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      r[k] = a[k] - b[k];
    
    return R;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> sub_ll_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = A[0][k] - B;

    return R;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> sub_rl_val(const Matrix<T1>& B, const T2 A) noexcept
  {
    Matrix<T3> R(B.size());

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = A - B[0][k];

    return R;
  }

  template<typename T1, typename T3>
  Matrix<T3> sub_ll_rng(const Matrix<T1>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    Matrix<T3> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = A[0][k] - uniform_distribution(generator);

    return R;
  }

  template<typename T1, typename T3>
  Matrix<T3> sub_rl_rng(const Matrix<T1>& B, const Random A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    Matrix<T3> R(B.size());

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = uniform_distribution(generator) - B[0][k];

    return R;
  }

  template<typename T>
  Matrix<T>& negate_inplace(Matrix<T>& A) noexcept
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] = -A[0][k];

    return A;
  }

  template<typename T>
  Matrix<T> negate(const Matrix<T> A)
  {
    Matrix<T> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
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
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator-(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return sub_mat_sequ(A, B);
  }
  
  template<typename T>
  Matrix<T> operator-(const Matrix<T>& A, const Matrix<T>& B)
  {
    #ifdef PARALLILOS_USE_PARALLELISM
      // return sub_mat_simd(A, B);
    #else
      return sub_mat_sequ(A, B);
    #endif
  }

  template<typename T, typename T3>
  Matrix<T3> operator-(const Matrix<T>& A, const Random B) noexcept
  {
    return sub_ll_rng(A, B);
  }

  template<typename T, typename T3>
  Matrix<T3> operator-(const Random A, const Matrix<T>& B) noexcept
  {
    return sub_rl_rng(B, A);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator-(const Matrix<T1>& A, const T2 B) noexcept
  {
    return sub_ll_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator-(const T1 A, const Matrix<T2>& B) noexcept
  {
    return sub_rl_val(B, A);
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator-(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(sub_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator-(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(sub_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
  {
    return std::move(sub_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
  {
    return std::move(sub_ll_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator-(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(sub_ll_mat_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator-(Matrix<T>&& A, const Random B) noexcept
  {
    return std::move(sub_ll_rng_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator-(const Random A, Matrix<T>&& B) noexcept
  {
    return std::move(sub_rl_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator-(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(sub_rl_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3>
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
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return A;
    }

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] /= B[0][k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& div_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return B;
    }

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      B[0][k] = A[0][k] / B[0][k];

    return B;
  }

  template<typename T1, typename T2>
  Matrix<T1>& div_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const T1 iB = 1.0 / B;

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] *= iB;

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& div_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept
  {
    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      B[0][k] = A / B[0][k];

    return B;
  }

  template<typename T>
  Matrix<T>& div_ll_rng_inplace(Matrix<T>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);
    
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] /= uniform_distribution(generator);

    return A;
  }

  template<typename T>
  Matrix<T>& div_rl_rng_inplace(Matrix<T>& B, const Random A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(A.min_, A.max_);

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      B[0][k] = uniform_distribution(generator) / B[0][k];

    return B;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> div_mat_sequ(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());

    const T1* a = A.data();
    const T2* b = B.data();
    T3* r = R.data();

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      r[k] = a[k] / b[k];
    
    return R;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> div_ll_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());
    const T1 iB = 1.0 / B;

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = A[0][k] * iB;

    return R;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> div_rl_val(const Matrix<T1>& B, const T2 A) noexcept
  {
    Matrix<T3> R(B.size());

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = A / B[0][k];

    return R;
  }

  template<typename T1, typename T3>
  Matrix<T3> div_ll_rng(const Matrix<T1>& A, const Random B) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(B.min_, B.max_);

    Matrix<T3> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = A[0][k] / uniform_distribution(generator);

    return R;
  }

  template<typename T1, typename T3>
  Matrix<T3> div_rl_rng(const Matrix<T1>& B, const Random A) noexcept
  {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(A.min_, A.max_);

    Matrix<T3> R(B.size());

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
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
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator/(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return div_mat_sequ(A, B);
  }
  
  template<typename T>
  Matrix<T> operator/(const Matrix<T>& A, const Matrix<T>& B)
  {
    #ifdef PARALLILOS_USE_PARALLELISM
      // return div_mat_simd(A, B);
    #else
      return div_mat_sequ(A, B);
    #endif
  }

  template<typename T, typename T3>
  Matrix<T3> operator/(const Matrix<T>& A, const Random B) noexcept
  {
    return div_ll_rng(A, B);
  }

  template<typename T, typename T3>
  Matrix<T3> operator/(const Random A, const Matrix<T>& B) noexcept
  {
    return div_rl_rng(B, A);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator/(const Matrix<T1>& A, const T2 B) noexcept
  {
    return div_ll_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator/(const T1 A, const Matrix<T2>& B) noexcept
  {
    return div_rl_val(B, A);
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator/(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(div_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(div_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
  {
    return std::move(div_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
  {
    return std::move(div_ll_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator/(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(div_ll_mat_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator/(Matrix<T>&& A, const Random B) noexcept
  {
    return std::move(div_ll_rng_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator/(const Random A, Matrix<T>&& B) noexcept
  {
    return std::move(div_rl_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator/(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(div_rl_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(div_ll_val_inplace(A, B));
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& pow_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return A;
    }

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] = std::pow(A[0][k], B[0][k]);

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& pow_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return B;
    }

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      B[0][k] = std::pow(A[0][k], B[0][k]);

    return B;
  }

  template<typename T1, typename T2>
  Matrix<T1>& pow_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] = std::pow(A[0][k], B);

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& pow_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept
  {
    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
      B[0][k] = std::pow(A, B[0][k]);

    return B;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> pow_mat(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return Matrix<T3>();
    }

    Matrix<T3> R(A.size());
  
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::pow(A[0][k], B[0][k]);

    return R;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> pow_ll_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::pow(A[0][k], B);

    return R;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> pow_rl_val(const Matrix<T1>& B, const T2 A) noexcept
  {
    Matrix<T3> R(B.size());

    const unsigned n = B.numel();
    for (unsigned k = 0; k < n; ++k)
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
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator^(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return pow_mat(A, B);
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator^(const Matrix<T1>& A, const T2 B) noexcept
  {
    return pow_ll_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator^(const T1 A, const Matrix<T2>& B) noexcept
  {
    return pow_rl_val(B, A);
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator^(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(pow_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(pow_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
  {
    return std::move(pow_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
  {
    return std::move(pow_ll_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator^(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(pow_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator^(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(pow_rl_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(pow_ll_val_inplace(A, B));
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> floor(const Matrix<T>& A)
  {
    static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
    Matrix<T> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::floor(A[0][k]);

    return R;
  }

  template<typename T>
  Matrix<T>&& floor(Matrix<T>&& A) noexcept
  {
    static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] = std::floor(A[0][k]);

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> round(const Matrix<T>& A)
  {
    static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
    Matrix<T> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::round(A[0][k]);

    return R;
  }

  template<typename T>
  Matrix<T>&& round(Matrix<T>&& A) noexcept
  {
    static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      A[0][k] = std::round(A[0][k]);
    }

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> ceil(const Matrix<T>& A)
  {
    static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
    Matrix<T> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      R[0][k] = std::ceil(A[0][k]);
    }

    return R;
  }

  template<typename T>
  Matrix<T>&& ceil(Matrix<T>&& A) noexcept
  {
    static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      A[0][k] = std::round(A[0][k]);
    }

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<double> mul(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    Matrix<double> result;

    if (A.N() != B.M())
    {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return result;
    }

    result = Matrix<double>(A.M(), B.N(), 0);

    for (unsigned i = 0; i < B.N(); i++)
    {
      for (unsigned j = 0; j < A.M(); j++)
      {
        for (unsigned k = 0; k < A.N(); k++)
        {
          result[j][i] += A[j][k] * B[k][i];
        }
      }
    }

    return result;
  }

  template<typename T>
  Matrix<T> div_sequ(const Matrix<T>& b, Matrix<T> A)
  {
    Matrix<double> x;

    // verify vertical dimensions
    if (b.M() != A.M()) 
    {
      PINAKAS_ERROR("vertical dimensions mismatch (b is %ux_, A is %ux_)", b.M(), A.M());
      return x;
    }

    // verify that b is a column matrix
    if (b.N() != 1)
    {
      PINAKAS_ERROR("b's horizontal dimension is not 1 (b is _x%u", b.N());
      return x;
    }

    // store the dimensions of A
    const unsigned M = A.M();
    const unsigned N = A.N();

    // QR decomposition matrices and result matrix
    Matrix<double> Q(M, N), R(N, N);
    x = Matrix<double>(N, 1);

    // QR decomposition using the modified Gram-Schmidt process
    for (unsigned i = 0; i < N; ++i)
    {
      // calculate the squared Euclidean norm of A's i'th column
      double sum_of_squares = 0;
      for (unsigned j = 0; j < M; ++j)
      {
        sum_of_squares += A[j][i] * A[j][i];
      }

      // skips if the squared Euclidean norm is 0
      if (sum_of_squares != 0)
      {
        // calculate the inverse Euclidean norm of A's i'th column
        double inorm = std::pow(sum_of_squares, -0.5);
        // normalize and store A's normalized i'th column
        for (unsigned j = 0; j < M; ++j)
        {
          Q[j][i] = A[j][i] * inorm;
        }
      }

      // orthogonalize the remaining columns with respects to A's i'th column
      for (unsigned k = i; k < N; ++k)
      {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        double projection = 0;
        for (unsigned j = 0; j < M; ++j)
        {
          projection += Q[j][i] * A[j][k];
        }

        // construct upper triangle matrix R using Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k >= i)
        {
          R[i][k] = projection;
        }

        // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k != i) // skips if k == i because the projection would be 0
        {
          for (unsigned j = 0; j < M; ++j)
          {
            A[j][k] -= projection * Q[j][i];
          }
        }
      }
    }

    // solve linear system Rx = Qt*b using back substitution
    for (unsigned i = N - 1; i < N; --i)
    {
      // calculate appropriate Qt*b component
      double substitution = 0;
      for (unsigned j = 0; j < M; ++j)
      {
        substitution += Q[j][i] * b[j][0];
      }

      // back substitution of previously solved x components
      for (unsigned k = N - 1; k > i; --k)
      {
        substitution -= R[i][k] * x[k][0];
      }

      // solve x's i'th component
      x[i][0] = substitution / R[i][i];
    }

    return x;
  }

  template<typename T>
  Matrix<T> div(const Matrix<T>& b, Matrix<T> A)
  {
    static_assert(std::is_convertible<T, double>::value, "T must be convertible to double");

    return div_sequ(b, A);
  }
// --------------------------------------------------------------------------------------
  template<template<typename> class M1, typename T>
  Matrix<T> transpose(const M1<T>& A)
  {
    Matrix<T> R(A.N(), A.M());
    for (unsigned y = 0; y < A.M(); ++y)
      for (unsigned x = 0; x < A.N(); ++x)
        R[x][y] = A[y][x];
    return R;
  }

  template<typename T>
  Matrix<T>&& transpose(Matrix<T>&& A)
  {
    std::swap(A.size_.M, A.size_.N);

    return std::move(A);
  }

  template<template<typename> class M1, typename T>
  Matrix<T> reshape(const M1<T>& A, const unsigned M, const unsigned N)
  {
    Matrix<T> R;

    if (A.numel() != M * N)
    {
      PINAKAS_ERROR("can't reshape %ux%u into %ux%u)", A.M(), A.N(), M, N);
      return R;
    }

    R = Matrix<T>(M, N);

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      R[0][k] = A[0][k];
    }

    return R;
  }

  template<typename T>
  Matrix<T>&& reshape(Matrix<T>&& A, const unsigned M, const unsigned N)
  {
    if (A.size_.numel != M*N)
    {
      PINAKAS_ERROR("can't reshape %ux%u into %ux%u)", A.M(), A.N(), M, N);
      goto exit;
    }

    A.size_.M = M;
    A.size_.N = N;

  exit:
    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<template<typename> class M1, typename T>
  T min(const M1<T>& A) noexcept
  {
    T minimum = A[0][0];
    for (unsigned k = 1; k < A.numel(); ++k)
      if (A[0][k] < minimum)
        minimum = A[0][k];
        
    return minimum;
  }
  
  template<template<typename> class M1, typename T>
  T max(const M1<T>& A) noexcept
  {
    T maximum = A[0][0];
    for (unsigned k = 1; k < A.numel(); ++k)
      if (A[0][k] > maximum)
        maximum = A[0][k];

    return maximum;
  }
  
  template<template<typename> class M1, typename T>
  T sum(const M1<T>& A) noexcept
  {
    T summation = 0;
    for (unsigned k = 0; k < A.numel(); ++k)
      summation += A[0][k];

    return summation;
  }

  template<template<typename> class M1, typename T>
  double prod(const M1<T>& A) noexcept
  {
    double temporary = 0;
    for (unsigned k = 0; k < A.numel(); ++k)
      temporary += std::log(A[0][k]);

    return std::exp(temporary);
  }

  template<template<typename> class M1, typename T>
  double avg(const M1<T>& A) noexcept
  {
    double average = 0;
    for (unsigned k = 0; k < A.numel(); ++k)
      average += A[0][k];

    return average/A.numel();
  }

  template<template<typename> class M1, typename T>
  double rms(const M1<T>& A) noexcept
  {
    const unsigned n = A.numel();
    double temporary = 0;
    for (unsigned k = 0; k < n; ++k)
      temporary += A[0][k] * A[0][k];
    
    return std::sqrt(temporary);
  }

  template<template<typename> class M1, typename T>
  double geo(const M1<T>& A) noexcept
  {
    double temporary = 0;
    for (unsigned k = 0; k < A.numel(); ++k)
      temporary += std::log(A[0][k]);

    return std::exp(temporary / A.numel());
  }
// --------------------------------------------------------------------------------------
  Matrix<double> orthogonalize(Matrix<double> A)
  {
    const unsigned M = A.M();
    const unsigned N = A.N();

    Matrix<double> Q(M, N);

    // matrix orthogonalization using the modified Gram-Schmidt process
    for (unsigned i = 0; i < N; ++i) {
      // calculate the squared Euclidean norm of A's i'th column
      double sum_of_squares = 0;
      for (unsigned j = 0; j < M; ++j)
        sum_of_squares += A[j][i] * A[j][i];

      // skips if the squared Euclidean norm is 0
      if (sum_of_squares != 0) {
        // calculate the inverse Euclidean norm of A's i'th column
        double inorm = std::pow(sum_of_squares, -0.5);
        // normalize and store A's normalized i'th column
        for (unsigned j = 0; j < M; ++j)
          Q[j][i] = A[j][i] * inorm;
      }

      // orthogonalize the remaining columns with respects to A's i'th column
      for (unsigned k = i; k < N; ++k) {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        double projection = 0;
        for (unsigned j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];

        // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k != i) // skips if k == i because the projection would be 0
          for (unsigned j = 0; j < M; ++j)
            A[j][k] -= projection * Q[j][i];
      }
    }

    return Q;
  }

  std::unique_ptr<Matrix<double>[]> qr(Matrix<double> A)
  {
    const unsigned M = A.M();
    const unsigned N = A.N();

    Matrix<double> Q(M, N);
    Matrix<double> R(N, N, 0);

    // QR decomposition using the modified Gram-Schmidt process
    for (unsigned i = 0; i < N; ++i) {
      // calculate the squared Euclidean norm of A's i'th column
      double sum_of_squares = 0;
      for (unsigned j = 0; j < M; ++j)
        sum_of_squares += A[j][i] * A[j][i];

      // skips if the squared Euclidean norm is 0
      if (sum_of_squares != 0) {
        // calculate the inverse Euclidean norm of A's i'th column
        double inorm = std::pow(sum_of_squares, -0.5);
        // normalize and store A's normalized i'th column
        for (unsigned j = 0; j < M; ++j)
          Q[j][i] = A[j][i] * inorm;
      }

      // orthogonalize the remaining columns with respects to A's i'th column
      for (unsigned k = i; k < N; ++k) {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        double projection = 0;
        for (unsigned j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];

        // construct upper triangle matrix R using Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k >= i)
          R[i][k] = projection;

        // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k != i) // skips if k == i because the projection would be 0
          for (unsigned j = 0; j < M; ++j)
            A[j][k] -= projection * Q[j][i];
      }
    }

    return std::unique_ptr<Matrix<double>[]>(new Matrix<double>[2]{std::move(Q), std::move(R)});
  }

  std::unique_ptr<Matrix<double>[]> linearize(const Matrix<double>& data_x, const Matrix<double>& data_y)
  {
    auto result = std::unique_ptr<Matrix<double>[]>(new Matrix<double>[2]);
    if (data_x.numel() != data_y.numel())
    {
      PINAKAS_ERROR("'data_x' and 'data_y' must have the same number of elements");
      return result;
    }

    if ((data_x.M() != 1) || (data_y.M() != 1))
    {
      PINAKAS_WARNING("data is interpreted as a horizontal 1-dimensional matrix");
    }

    const unsigned n = data_x.numel();

    Matrix<double> lin_x(1, n, 0);
    Matrix<double> lin_y(1, n, 0);

    // set first and last values of the linearized data set
    lin_x[0][0]   = data_x[0][0];
    lin_y[0][0]   = data_y[0][0];
    lin_x[0][n-1] = data_x[0][n-1];
    lin_y[0][n-1] = data_y[0][n-1];

    // build linearly spaced x data and its associated y value
    const double step = (data_x[0][n - 1] - data_x[0][0]) / (n - 1);
    for (unsigned k = 1; k < (n - 1); ++k)
    {
      // build linearly spaced x data
      lin_x[0][k] = lin_x[0][k - 1] + step;

      // linearly interpolate y value
      double x1 = data_x[0][k];
      double y1 = data_y[0][k];
      double x2 = data_x[0][k+1];
      double y2 = data_y[0][k+1];
      lin_y[0][k] = y1 + (lin_x[0][k] - x1) * (y2 - y1) / (x2 - x1);
    }

    result[0] = std::move(lin_x);
    result[1] = std::move(lin_y);
    return result;
  }


  template<template<typename> class M>
  std::unique_ptr<M<double>[]> linearize(M<double>&& data_x, M<double>&& data_y)
  {
    auto result = std::unique_ptr<M<double>[]>(new M<double>[2]);
    if (data_x.numel() != data_y.numel())
    {
      PINAKAS_ERROR("'data_x' and 'data_y' must have the same number of elements");
      return result;
    }

    if ((data_x.M() != 1) || (data_y.M() != 1))
    {
      PINAKAS_WARNING("data is interpreted as a horizontal 1-dimensional matrix");
    }

    const unsigned n = data_x.numel();

    // build linearly spaced x data and its associated y value
    const double step = (data_x[0][n - 1] - data_x[0][0]) / (n - 1);
    for (unsigned k = 1; k < (n - 1); ++k) {
      // store necessary data to linearly interpolate y value
      double x1 = data_x[0][k];
      double y1 = data_y[0][k];
      double x2 = data_x[0][k+1];
      double y2 = data_y[0][k+1];

      // build linearly spaced data
      data_x[0][k] = data_x[0][k - 1] + step;
      data_y[0][k] = y1 + (data_x[0][k] - x1) * (y2 - y1) / (x2 - x1);
    }

    result[0] = std::move(data_x);
    result[1] = std::move(data_y);
    return result;
  }

  template<typename T>
  Matrix<T> linspace(const double x1, const double x2, const unsigned N)
  {
    Matrix<T> R(1, N);

    const unsigned n = N - 1;
    const double step  = (x2 - x1) / (N - 1);

    double temporary = x1;
    for (unsigned k = 1; k < n; ++k)
    {
      // R[0][k] = std::round(temporary += step);
      R[0][k] = temporary += step;
    }

    R[0][0] = x1;
    R[0][n] = x2;

    return R;
  }

  template<typename T>
  Matrix<T> iota(const unsigned n)
  {
    Matrix<T> R(1, n);

    for (unsigned k = 0; k < n; ++k)
      R[0][k] = k;

    return R;
  }

  template<typename T>
  Matrix<T> eye(const unsigned M, const unsigned N)
  {
    Matrix<T> R(M, N, 0);

    const unsigned n = std::min(M, N);

    for (unsigned k = 0; k < n; ++k)
      R[k][k] = 1;
    
    return R;
  }

  Matrix<double> diff(const Matrix<double>& A, unsigned n)
  {
    if (n) {
      Matrix<double> R(A.M(), A.N() - 1);
      for (unsigned y = 0; y < R.M(); ++y)
        for (unsigned x = 0; x < R.N(); ++x)
          R[y][x] = A[y][x + 1] - A[y][x];

      return diff(R, n - 1);
    }
    return A;
  }

  template<typename T>
  Matrix<T> reverse(const Matrix<T>& A)
  {
    Matrix<T> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      R[0][k] = A[0][n-1 - k];
    }

    return R;
  }

  template<template<typename> class M1, typename T>
  M1<T>&& reverse(M1<T>&& A) noexcept
  {
    const unsigned n   = A.numel();
    const unsigned n_2 = n >> 1;
    for (unsigned k = 0; k < n_2; ++k)
    {
      std::swap(A[0][k], A[0][n-1 - k]);
    }

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2, typename T3>
  Matrix<T3> conv(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    const unsigned n1 = A.numel();
    const unsigned n2 = B.numel();

    Matrix<T3> convoluted(1, n1 + n2 - 1, 0);
    for (unsigned i = 0; i < n1; ++i)
      for (unsigned j = 0; j < n2; ++j)
        convoluted[0][i + j] += A[0][i] * B[0][j];

    return convoluted;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> corr(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    const unsigned n1 = A.numel();
    const unsigned n2 = B.numel();

    Matrix<T3> result(1, n1 + n2 - 1, 0);
    for (unsigned i = 0; i < n1; ++i)
      for (unsigned j = 0; j < n2; ++j)
        result[0][i + j] += A[0][n1-1 - i] * B[0][j];

    return result; 
  }

  template<typename T>
  Matrix<T> corr(const Matrix<T>& A)
  {
    const unsigned n = A.numel();

    Matrix<T> result(1, 2*n - 1, 0);
    for (unsigned i = 0; i < n; ++i)
      for (unsigned j = 0; j < n; ++j)
        result[0][i + j] += A[0][n-1 - i] * A[0][j];

    return result;
  }

  Matrix<double> rxx(const Matrix<double>& A)
  {
    const unsigned n = A.numel();

    Matrix<double> R(1, n, 0);
    for (unsigned i = 0; i < n; ++i)
      for (unsigned j = 0; j < n; ++j)
        if ((i+j - n+1) < n)
          R[0][i+j - n+1] += A[0][n-1 - i] * A[0][j];

    return R;
  }

  Matrix<double> rxx(const Matrix<double>& A, const unsigned K)
  {
    const unsigned n = A.numel();

    Matrix<double> R(1, K, 0);
    for (unsigned i = 0; i < n; ++i)
      for (unsigned j = 0; j < n; ++j)
        if ((i+j - n+1) < K)
          R[0][i+j - n+1] += A[0][n-1 - i] * A[0][j];

    return R;
  }

  Matrix<double> lpc(const Matrix<double>& A, const unsigned p)
  {
    Matrix<double> result;

    if (p >= A.numel())
    {
      PINAKAS_ERROR("'p' should be less than number of elements in 'A'");
      return result;
    }

    if (A.M() != 1)
    {
      PINAKAS_WARNING("'A' should be a horizontal vector");
    }

    Matrix<double> rxx_ = rxx(A, p+1);

    Matrix<double> autocorr_mat(p, p);
    Matrix<double> autocorr_vec(p, 1);
    for (unsigned i = 0; i < p; ++i)
    {
      for (unsigned j = 0; j < p; ++j)
      {
        autocorr_mat[j][i] = rxx_[0][(j > i) ? (j - i) : (i - j)];
      }

      autocorr_vec[0][i] = rxx_[0][i+1];
    }

    result = div(autocorr_vec, autocorr_mat);

    return result;
  }

  Matrix<double> toeplitz(const Matrix<double>& A)
  {
    const unsigned n = A.numel();
    Matrix<double> result(n, n);
    for (unsigned i = 0; i < n; ++i)
    {
      for (unsigned j = 0; j < n; ++j)
      {
        result[j][i] = A[0][(j > i) ? (j - i) : (i - j)];
      }
    }

    return result;
  }
// --------------------------------------------------------------------------------------
  Matrix<double> blackman(const unsigned N)
  {
    Matrix<double> window(1, N);
    for (unsigned k = 0; k < N; ++k)
    {
      const double temporary = (2 * M_PI * k)/(N - 1);
      window[0][k] = 0.42 - 0.50 * std::cos(temporary) + 0.08 * std::cos(2 * temporary);
    }

    return window;
  }

  Matrix<double> blackman(const Matrix<double>& signal)
  {
    const unsigned N = signal.numel();
    Matrix<double> windowed(1, N);
    for (unsigned k = 0; k < N; ++k)
    {
      const double temporary = (2 * M_PI * k)/(N - 1);
      windowed[0][k] = signal[0][k] * (0.42 - 0.50 * std::cos(temporary) + 0.08 * std::cos(2 * temporary));
    }

    return windowed;
  }

  Matrix<double>&& blackman(Matrix<double>&& signal) noexcept
  {
    const unsigned N = signal.numel();
    for (unsigned k = 0; k < N; ++k)
    {
      const double temporary = (2 * M_PI * k)/(N - 1);
      signal[0][k] *= 0.42 - 0.50 * std::cos(temporary) + 0.08 * std::cos(2 * temporary);
    }

    return std::move(signal);
  }
// --------------------------------------------------------------------------------------
  Matrix<double> hamming(const unsigned N)
  {
    Matrix<double> window(1, N);
    for (unsigned k = 0; k < N; ++k)
    {
      window[0][k] = 0.54 - 0.46 * std::cos(2 * M_PI * k / (N - 1));
    }

    return window;
  }

  Matrix<double> hamming(const Matrix<double>& signal)
  {
    const unsigned N = signal.numel();
    Matrix<double> windowed(1, N);
    for (unsigned k = 0; k < N; ++k)
    {
      windowed[0][k] = signal[0][k] * (0.54 - 0.46 * std::cos(2 * M_PI * k / (N - 1)));
    }

    return windowed;
  }

  Matrix<double>&& hamming(Matrix<double>&& signal) noexcept
  {
    const unsigned n = signal.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      signal[0][k] *= 0.54 - 0.46 * std::cos(2 * M_PI * k / (n - 1));
    }

    return std::move(signal);
  }
// --------------------------------------------------------------------------------------
  Matrix<double> hann(const unsigned N)
  {
    Matrix<double> window(1, N);
    for (unsigned k = 0; k < N; ++k)
    {
      window[0][k] = 0.5 - 0.5 * std::cos(2 * M_PI * k / (N - 1));
    }

    return window;
  }

  Matrix<double> hann(const Matrix<double>& signal)
  {
    const unsigned N = signal.numel();
    Matrix<double> windowed(1, N);
    for (unsigned k = 0; k < N; ++k)
    {
      windowed[0][k] = signal[0][k] * (0.5 - 0.5 * std::cos(2 * M_PI * k / (N - 1)));
    }

    return windowed;
  }

  Matrix<double>&& hann(Matrix<double>&& signal) noexcept
  {
    const unsigned N = signal.numel();
    for (unsigned k = 0; k < N; ++k)
    {
      signal[0][k] *= 0.5 - 0.5 * std::cos(2 * M_PI * k / (N - 1));
    }

    return std::move(signal);
  }
// --------------------------------------------------------------------------------------
  double newton(const std::function<double(double)> function,
                const double tol,
                const unsigned max_iteration,
                const double seed) noexcept
  {
    const double half_tol = tol * 0.5;

    double root = seed;

    unsigned iteration = 0;
    while ((tol < std::abs(function(root))) && (iteration++ < max_iteration))
    {
      root -= tol * function(root) / (function(root + half_tol) - function(root - half_tol));
    }

    return root;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<double> cos(Matrix<T>& A)
  {
    Matrix<double> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::cos(A[0][k]);

    return R;
  }

  Matrix<double>&& cos(Matrix<double>&& A) noexcept
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] = std::cos(A[0][k]);

    return std::move(A);
  }

  template<typename T>
  Matrix<double> sin(Matrix<T>& A)
  {
    Matrix<double> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::sin(A[0][k]);

    return R;
  }

  Matrix<double>&& sin(Matrix<double>&& A) noexcept
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] = std::sin(A[0][k]);

    return std::move(A);
  }

  template<typename T>
  Matrix<double> sinc(Matrix<T>& A)
  {
    Matrix<double> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      double temporary = M_PI * A[0][k];
      R[0][k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return R;
  }

  Matrix<double>&& sinc(Matrix<double>&& A) noexcept
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
    {
      double temporary = M_PI * A[0][k];
      A[0][k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  Matrix<double> upsample(const Matrix<double>& data, const unsigned L)
  {
    Matrix<double> upsampled(1, L * data.numel(), 0);

    for (unsigned k = 0; k < data.numel(); ++k)
    {
      upsampled[0][k * L] = data[0][k];
    }

    return upsampled;
  }

  Matrix<double> sinc_fir(const unsigned length, const double frequency)
  {
    Matrix<double> impulse;

    // validate impulse length
    if ((length % 2) == 0)
    {
      PINAKAS_ERROR("'length' must be odd");
      return impulse;
    }

    // offset to the impulse center
    const signed int offset = (length - 1) * 0.5;

    // compute impulse
    impulse = Matrix<double>(1, length);
    for (signed int k = 0; k < signed(length); ++k)
    {
      double temporary = M_PI * (k - offset) * frequency;
      impulse[0][k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return impulse;
  }

  template<typename T>
  Matrix<double> resample(const Matrix<T>& data, const unsigned L, const unsigned keep, const double alpha, const bool tail)
  {
    static_assert(std::is_convertible<T, double>::value, "T must be convertible to double");
    Matrix<double> resampled;

    if (data.N() < 2)
    {
      PINAKAS_ERROR("'data' must be atleast 2 element wide");
      return resampled;
    }
    
    if (L <= 1)
    {
      PINAKAS_ERROR("'L' must be at least 2");
      return resampled;
    }
    
    if (keep >= data.numel())
    {
      PINAKAS_ERROR("'keep' must be less than the number of elements in 'data'");
      return resampled;
    }
    
    if (alpha < (1.0/L))
    {
      PINAKAS_ERROR("'alpha' must be at least 1/L");
      return resampled;
    }

    // design low-pass interpolation filter
    const unsigned offset        = L * alpha; // offset to impulse center
    const unsigned filter_length = 2 * offset + 1;
    const Matrix<double> filter  = blackman(sinc_fir(filter_length, 1.0 / L));

    const unsigned N = data.N();
    const unsigned M = data.M();

    // indices to the first and last upsampled elements in the symetrically extended data
    const unsigned first = L*keep;
    const unsigned last  = L*N + first - (tail ? 1 : L);

    // symetrically extended data
    const unsigned extended_length = N + 2 * keep;
    Matrix<double> extended(M, extended_length, 0);

    // resampled data
    resampled = Matrix<double> (M, last - first + 1, 0);
    
    // resample every row separately
    for (unsigned m = 0; m < M; ++m)
    {
      // store and upsample left symetrical data
      unsigned k = 0;
      for (unsigned i = 0; i < keep; ++i)
      {
        extended[m][k++] = 2*data[m][0] - data[m][keep - i];
      }

      // store and upsample data
      for (unsigned i = 0; i < N; ++i)
      {
        extended[m][k++] = data[m][i];
      }
        
      // store and upsample right symetrical data
      for (unsigned i = 0; i < keep; ++i)
      {        
        extended[m][k++] = 2*data[m][N-1] - data[m][N-2 - i];
      }

      // interpolate upsampled data using a cropped convolution
      for (unsigned i = 0; i < extended_length; ++i)
      {
        for (unsigned j = 0; j < filter_length; ++j)
        {
          unsigned k = i*L + j - offset;
          // skips if the index is not within the upsampled data range (cropping)
          if ((first <= k) && (k <= last))
          {
            resampled[m][k - first] += extended[m][i] * filter[m][j];
          }
        }
      }
    }

    return resampled;
  }

  template<template<typename> class M1, typename T>
  std::ostream& operator<<(std::ostream& ostream, const M1<T>& A)
  {
    if (A.numel() != 0)
    {
      unsigned max_len = 0;
      for (unsigned y = 0; y < A.M(); ++y)
      {
        for (unsigned x = 0; x < A.N(); ++x)
        {
          std::stringstream ss;
          ss.copyfmt(ostream);
          ss << A[y][x];
          max_len = std::max(size_t(max_len), ss.str().length());
        }
      }
      
      for (unsigned y = 0; y < A.M(); ++y)
      {
        for (unsigned x = 0; x < A.N(); ++x)
        {
          ostream << std::setw(max_len + 1) << A[y][x];
        }

        ostream << '\n';
      }
    }

    return ostream;
  }

  std::ostream& operator<<(std::ostream& ostream, const Size size)
  {
    return ostream << size.M << 'x' << size.N;
  }

  __attribute__((optimize("O0")))
  void plot(List<std::string> titles, List<DataSet> data_sets, bool persistent, bool remove, bool pause, bool lines)
  {
    // validate that gnuplot is in system path
    static bool gnuplot_on_system_path = false;
    if ((gnuplot_on_system_path == false) && (std::system("gnuplot --version") != 0))
    {
      PINAKAS_ERROR("gnuplot could not be found in the system path");
      return;
    }
    gnuplot_on_system_path = true;

    // validate that x and y have the same number of elements
    for (auto& data_set : data_sets)
    {
      const Matrix<double>& xdata = data_set.first;
      const Matrix<double>& ydata = data_set.second;
      if (xdata.numel() != ydata.numel())
      {
        PINAKAS_ERROR("number of element mismatch (x has %u elements, y has %u elements)", xdata.numel(), ydata.numel());
        return;
      }
    }

    if (titles.size() != data_sets.size())
    {
      PINAKAS_WARNING("number of titles does not equal number of data sets");
    }

    // create temporary file
    std::ofstream file{"gnuplot.txt"};

    // validate file opening
    if (file.fail())
    {
      PINAKAS_ERROR("could not create/open gnuplot data file");
      return;
    }

    // write x and y data to file for each data set
    for (const auto& data_set : data_sets)
    {
      const auto& x = data_set.first;
      const auto& y = data_set.second;

      for (unsigned k = 0; k < x.numel(); ++k)
      {
        file << x[0][k] << ' ' << y[0][k] << '\n';
      }

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
    {
      gnuplot_pipeline << " -persistent";
    }

    // plot all data sets
    gnuplot_pipeline << " -e \"set title \\\"gnuplot\\\"; plot 'gnuplot.txt'";
    unsigned k = 0;
    for (const std::string& title : titles) {
      if (k)
        gnuplot_pipeline << ", ''";
      gnuplot_pipeline << " index " << k;
      if (lines)
        gnuplot_pipeline << " with lines";
      gnuplot_pipeline << " title '" << title.c_str() << '\'';
      k++;
    }
    gnuplot_pipeline << '"';

    // conditionally pause after plotting
    if (pause)
    {
      gnuplot_pipeline << " -e \"pause -1 'press any key to continue...'\"";
    }

    // execute command pipeline
    std::system(gnuplot_pipeline.str().c_str());

    // conditionally remove file after creation
    if (remove)
    {
      std::remove("gnuplot.txt");
    }
    
    return;
  }
  
  template<typename T>
  Matrix<double> abs(const Matrix<T>& A)
  {
    Matrix<double> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::abs(A[0][k]);

    return R;
  }
  
  Matrix<double> real(const Matrix<complex>& A)
  {
    Matrix<double> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::real(A[0][k]);

    return R;
  }
  
  Matrix<double> imag(const Matrix<complex>& A)
  {
    Matrix<double> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::imag(A[0][k]);

    return R;
  }

  Matrix<complex> conj(const Matrix<complex>& A)
  {
    Matrix<complex> R(A.size());

    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      R[0][k] = std::conj(A[0][k]);

    return R;
  }

  Matrix<complex>&& conj(Matrix<complex>&& A)
  {
    const unsigned n = A.numel();
    for (unsigned k = 0; k < n; ++k)
      A[0][k] = std::conj(A[0][k]);

    return std::move(A);
  }

  Matrix<complex> fft(const Matrix<complex>& signal)
  {
    return fft(Matrix<complex>(signal));
  }

  Matrix<complex>&& fft(Matrix<complex>&& signal)
  {
    const unsigned N = signal.numel();

    if (N & (N - 1))
    {
      PINAKAS_ERROR("the number of elements in 'signal' must be a power of 2");
      return std::move(signal);
    }
    
    unsigned k = N; // Current stage size
    unsigned n; // Size of butterfly operations
    double thetaT = M_PI / N; // Angle for twiddle factor
    complex phiT = complex(std::cos(thetaT), -std::sin(thetaT)); // Twiddle factor for the first stage

    // radix-2 decimation-in-frequency variation of the Cooley-Tukey fft algorithm
    while (k > 1)
    {
      n = k;
      k *= 0.5; // halve stage size
      phiT *= phiT; // Square the twiddle factor for the next stage
      complex twiddle_factor = 1; // Initialize the twiddle factor for the current stage

      // butterfly operations
      for (unsigned l = 0; l < k; ++l)
      {
        for (unsigned a = l; a < N; a += n)
        {
          unsigned b = a + k;
          complex temporary = signal[0][a] - signal[0][b];
          signal[0][a] += signal[0][b];
          signal[0][b]  = temporary * twiddle_factor;
        }
        // Update the twiddle factor for the next butterfly operation
        twiddle_factor *= phiT;
      }
    }

    // re-order frequency bins
    const unsigned bits_to_reverse = std::log2(N);
    for (unsigned a = 0; a < N; ++a)
    {
      // b = bit reversal of a
      unsigned b = a;
      b = ((b & 0xAAAAAAAA) >> 1) | ((b & 0x55555555) << 1);
      b = ((b & 0xCCCCCCCC) >> 2) | ((b & 0x33333333) << 2);
      b = ((b & 0xF0F0F0F0) >> 4) | ((b & 0x0F0F0F0F) << 4);
      b = ((b & 0xFF00FF00) >> 8) | ((b & 0x00FF00FF) << 8);
      b = ((b >> 16) | (b << 16)) >> (32 - bits_to_reverse);
      
      // swap elements
      if (b > a)
      {
        std::swap(signal[0][a], signal[0][b]);
      }
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
}
