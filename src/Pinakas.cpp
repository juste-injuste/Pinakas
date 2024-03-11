// --inclusion guard---------------------------------------------------------------------
#include "../include/Pinakas.hpp"

#ifndef  M_PI
# define M_PI 3.14159265358979323846
#endif
// --Pinakas library: backend forward declaration----------------------------------------
namespace Pinakas
{
  namespace _backend
  {
    template<typename T1, typename T2>
    auto _add_mat_inplace(Matrix<T1>& A_, const Matrix<T2>& B_) -> Matrix<T1>&
    {
      if ((A_.M() != B_.M()) || (A_.N() != B_.N()))
      {
        PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A_.M(), A_.N(), B_.M(), B_.N());
        return A_;
      }

      const size_t n = A_.numel();
      for (size_t k = 0; k < n; ++k)
      {
        A_.data()[k] += B_.data()[k];
      }

      return A_;
    }

    template<typename T1, typename T2>
    auto _add_mat(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()+T2())>
    {
      Matrix<decltype(T1()+T2())> result;
      if ((A.M() != B.M()) || (A.N() != B.N()))
      {
        PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
        return result;
      }

      result = Matrix<decltype(T1()+T2())>(A.M(), A.N());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] + B.data()[k];
      }
      
      return result;
    }

    template<typename T1, typename T2>
    auto _add_val_inplace(Matrix<T1>& A, const T2 B) noexcept -> Matrix<T1>&
    {
      for (T1& data : A)
      {
        data += B;
      }

      return A;
    }

    template<typename T1, typename T2>
    auto _add_val(const Matrix<T1>& A, const T2 B) noexcept -> Matrix<decltype(T1()+T2())>
    {
      Matrix<decltype(T1()+T2())> result(A.size());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] + B;
      }

      return result;
    }

    template<typename T>
    auto _add_rng_inplace(Matrix<T>& A, Random B) noexcept -> Matrix<T>&
    {
      static std::random_device device;
      static std::mt19937 generator(device());
      std::uniform_real_distribution<float> uniform_distribution(B.min_, B.max_ + std::is_integral<T>::value);

      for (T& data : A)
      {
        data += uniform_distribution(generator);
      }

      return A;
    }

    template<typename T>
    auto _add_rng(const Matrix<T>& A, Random B) noexcept -> Matrix<T>
    {
      static std::random_device device;
      static std::mt19937 generator(device());
      std::uniform_real_distribution<float> uniform_distribution(B.min_, B.max_ + std::is_integral<T>::value);
      
      Matrix<T> result(A.size());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] + uniform_distribution(generator);
      }

      return result;
    }
  
    template<typename T1, typename T2>
    auto _mul_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&
    {
      if ((A.M() != B.M()) || (A.N() != B.N()))
      {
        PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
        return A;
      }

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        A.data()[k] *= B.data()[k];
      }

      return A;
    }

    template<typename T1, typename T2>
    auto _mul_mat(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()+T2())>
    {
      Matrix<decltype(T1()+T2())> result;
      if ((A.M() != B.M()) || (A.N() != B.N()))
      {
        PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
        return result;
      }

      result = Matrix<decltype(T1()+T2())>(A.M(), A.N());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] * B.data()[k];
      }
      
      return result;
    }

    template<typename T1, typename T2>
    auto _mul_val_inplace(Matrix<T1>& A, const T2 B) noexcept -> Matrix<T1>&
    {
      for (T1& data : A)
      {
        data *= B;
      }

      return A;
    }

    template<typename T1, typename T2>
    auto _mul_val(const Matrix<T1>& A, const T2 B) noexcept -> Matrix<decltype(T1()+T2())>
    {
      Matrix<decltype(T1()+T2())> result(A.size());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] * B;
      }

      return result;
    }

    template<typename T>
    auto _mul_rng_inplace(Matrix<T>& A, Random B) noexcept -> Matrix<T>&
    {
      static std::random_device device;
      static std::mt19937 generator(device());
      std::uniform_real_distribution<float> uniform_distribution(B.min_, B.max_ + std::is_integral<T>::value);

      for (T& data : A)
      {
        data *= uniform_distribution(generator);
      }

      return A;
    }

    template<typename T>
    auto _mul_rng(const Matrix<T>& A, Random B) noexcept -> Matrix<T>
    {
      static std::random_device device;
      static std::mt19937 generator(device());
      std::uniform_real_distribution<float> uniform_distribution(B.min_, B.max_ + std::is_integral<T>::value);
      
      Matrix<T> result(A.size());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] * uniform_distribution(generator);
      }

      return result;
    }

    template<typename T>
    auto _neg_inplace(Matrix<T>& A) noexcept -> Matrix<T>&
    {
      for (T& data : A)
      {
        data = -data;
      }

      return A;
    }

    template<typename T1, typename T2>
    auto _sub_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&
    {
      if ((A.M() != B.M()) || (A.N() != B.N()))
      {
        PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
        return A;
      }

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        A.data()[k] -= B.data()[k];
      }

      return A;
    }

    template<typename T1, typename T2>
    auto _sub_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A) -> Matrix<T1>&
    {
      if ((A.M() != B.M()) || (A.N() != B.N()))
      {
        PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
        return B;
      }

      const size_t n = B.numel();
      for (size_t k = 0; k < n; ++k)
      {
        B.data()[k] = A.data()[k] - B.data()[k];
      }

      return B;
    }

    template<typename T1, typename T2>
    auto _sub_ll_val_inplace(Matrix<T1>& A, T2 B) noexcept -> Matrix<T1>&
    {
      for (T1& data : A)
      {
        data -= B;
      }

      return A;
    }

    template<typename T1, typename T2>
    auto _sub_rl_val_inplace(Matrix<T1>& B, T2 A) noexcept -> Matrix<T1>&
    {
      for (T1& data : B)
      {
        data = A - data;
      }

      return B;
    }

    template<typename T>
    auto _sub_ll_rng_inplace(Matrix<T>& A, Random B) noexcept -> Matrix<T>&
    {
      static std::random_device device;
      static std::mt19937 generator(device());
      std::uniform_real_distribution<float> uniform_distribution(B.min_, B.max_ + std::is_integral<T>::value);

      for (T& data : A)
      {
        data -= uniform_distribution(generator);
      }

      return A;
    }

    template<typename T>
    auto _sub_rl_rng_inplace(Matrix<T>& B, Random A) noexcept -> Matrix<T>&
    {
      static std::random_device device;
      static std::mt19937 generator(device());
      std::uniform_real_distribution<float> uniform_distribution(A.min_, A.max_ + std::is_integral<T>::value);

      for (T& data : B)
      {
        data = uniform_distribution(generator) - data;
      }

      return B;
    }

    template<typename T>
    auto _neg_mat(const Matrix<T> A) -> Matrix<T>
    {
      Matrix<T> result(A.size());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = -A.data()[k];
      }

      return result;
    }

    template<typename T1, typename T2>
    auto _sub_mat(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()-T2())>
    {
      Matrix<decltype(T1()-T2())> result;

      if ((A.M() != B.M()) || (A.N() != B.N()))
      {
        PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
        return result;
      }

      result = Matrix<decltype(T1()-T2())>(A.M(), A.N());
      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] - B.data()[k];
      }
      
      return result;
    }

    template<typename T1, typename T2>
    auto _sub_ll_val(const Matrix<T1>& A, T2 B) noexcept -> Matrix<decltype(T1()-T2())>
    {
      Matrix<decltype(T1()-T2())> result(A.size());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] - B;
      }

      return result;
    }

    template<typename T1, typename T2>
    auto _sub_rl_val(const Matrix<T1>& B, T2 A) noexcept -> Matrix<decltype(T1()-T2())>
    {
      Matrix<decltype(T1()-T2())> result(B.size());

      const size_t n = B.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A - B.data()[k];
      }

      return result;
    }

    template<typename T>
    auto _sub_ll_rng(const Matrix<T>& A, Random B) noexcept -> Matrix<decltype(T()-float())>
    {
      static std::random_device device;
      static std::mt19937 generator(device());
      std::uniform_real_distribution<float> uniform_distribution(B.min_, B.max_ + std::is_integral<T>::value);

      Matrix<decltype(T()-float())> result(A.size());

      const size_t n = A.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = A.data()[k] - uniform_distribution(generator);
      }

      return result;
    }

    template<typename T>
    auto _sub_rl_rng(const Matrix<T>& B, Random A) noexcept -> Matrix<decltype(T()-float())>
    {
      static std::random_device device;
      static std::mt19937 generator(device());
      std::uniform_real_distribution<float> uniform_distribution(A.min_, A.max_ + std::is_integral<T>::value);

      Matrix<decltype(T()-float())> result(B.size());

      const size_t n = B.numel();
      for (size_t k = 0; k < n; ++k)
      {
        result.data()[k] = uniform_distribution(generator) - B.data()[k];
      }

      return result;
    }
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
    delete[] _data;
    PINAKAS_LOG("deleted %ux%u", _size.M, _size.N);
  }

  template<typename T>
  Matrix<T>::Matrix() noexcept :
    _size{0, 0, 0},
    _true_size{0, 0, 0},
    _data(nullptr)
  {
    PINAKAS_LOG("created %ux%u", _size.M, _size.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const Matrix<T>& other)
  {
    if (this != &other)
    {
      // allocate memory
      _allocate(other._size.M, other._size.N);

      // store value
      for (size_t k = 0; k < _size.numel; ++k)
      {
        _data[k] = other.data()[k];
      }
    }
    else PINAKAS_ERROR("could not copy assign, self reference is not supported");

    PINAKAS_LOG("copied %ux%u", other._size.M, other._size.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const Slice<T>& other)
  {
    if (this != static_cast<const void*>(&other))
    {
      // allocate memory
      _allocate(other.M(), other.N());

      // store value
      for (size_t m = 0; m < _size.M; ++m)
      {
        for (size_t n = 0; n < _size.N; ++n)
        {
          *this[m][n] = other[m][n];
        }
      }
    }
    else PINAKAS_ERROR("could not copy assign, self reference is not supported");

    PINAKAS_LOG("copied %ux%u", other.size_.M, other.size_.N);
  }

  template<typename T>
  Matrix<T>::Matrix(Matrix<T>&& other) noexcept :
    _size(other._size),
    _data(other._data)
  {
    other._size      = Size{0, 0, 0};
    other._true_size = other._size;
    other._data      = nullptr;

    PINAKAS_LOG("moved %ux%u", _size.M, _size.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const size_t M, const size_t N)
  {
    // allocate memory
    _allocate(M, N);

    PINAKAS_LOG("created %ux%u", _size.M, _size.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const Size size) :
    Matrix(size.M, size.N)
  {}

  template<typename T>
  Matrix<T>::Matrix(const size_t M, const size_t N, const T value)
  {
    // allocate memory
    _allocate(M, N);

    // store values
    for (auto& data : *this)
    {
      data = value;
    }

    PINAKAS_LOG("created %ux%u and filled", _size.M, _size.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const Size size, T value)
    : Matrix(size.M, size.N, value)
  {}

  template<typename T>
  Matrix<T>::Matrix(const size_t M, const size_t N, Random range)
  {
    // allocate memory
    _allocate(M, N);

    // random number generator
    static std::random_device device;
    static std::mt19937 generator(device());
    std::uniform_real_distribution<float> uniform_distribution(range.min_, range.max_ + std::is_integral<T>::value);

    // assign random value to matrix
    for (size_t k = 0; k < _size.numel; ++k)
    {
      _data[k] = uniform_distribution(generator);
    }
      
    PINAKAS_LOG("created %ux%u from range [%f, %f]", M, N, range.min_, range.max_ + std::is_integral<T>::value);
  }

  template<typename T>
  Matrix<T>::Matrix(const Size size, Random range) :
    Matrix(size.M, size.N, range)
  {}

  template<typename T>
  Matrix<T>::Matrix(const List<T> list)
  {
    // allocate memory
    _allocate(1, list.size());

    // store values
    size_t x = 0;
    for (T value : list)
    {
      _data[x++] = value;
    }

    PINAKAS_LOG("created %ux%u", _size.M, _size.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const List<const List<const T>> values)
  {
    // dimension validation
    size_t temp_N = 0;
    for (const List<const T>& vector : values)
    {
      if (temp_N && (temp_N != vector.size()))
      {
        PINAKAS_ERROR("vertical dimensions mismatch (%u vs %u)", temp_N, size_t(vector.size()));
        _size      = Size{0, 0, 0};
        _true_size = _size;
        return;
      }
      else
      {
        temp_N = vector.size();
      }
    }

    // allocate memory
    _allocate(values.size(), temp_N);

    // store values
    size_t y = 0;
    for (const auto& vector : values)
    {
      size_t x = 0;
      for (T value : vector)
      {
        _data[x + y * _size.N] = value;
        ++x;
      }
      ++y;
    }

    PINAKAS_LOG("created %ux%u", _size.M, _size.N);
  }

  template<typename T>
  Matrix<T>::Matrix(const List<const Matrix<T>> list)
  {
    // dimension validation
    size_t M_ = 0;
    size_t N_ = 0;
    for (const Matrix<T>& matrix : list)
    {
      if (M_ && (M_ != matrix._size.M))
      {
        PINAKAS_ERROR("horizontal dimensions mismatch (%zu vs %zu)", M_, matrix._size.M);
        _size      = Size{0, 0, 0};
        _true_size = _size;
        return;
      }
      else
      {
        M_ = matrix._size.M;
      }

      N_ += matrix._size.N;
    }

    // allocate memory
    _allocate(M_, N_);

    // store values
    size_t k = 0;
    for (const Matrix<T>& matrix : list)
    {
      for (size_t y = 0; y < matrix._size.M; ++y)
      {
        for (size_t x = 0; x < matrix._size.N; ++x)
        {
          _data[x + k + y * _size.N] = matrix[y][x];
        }
      }

      k += matrix._size.N;
    }

    PINAKAS_LOG("created %ux%u from concatonation", _size.M, _size.N);
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  void Matrix<T>::_allocate(const size_t M, const size_t N)
  {
    // validate sizes
    if ((M == 0) || (N == 0))
    {
      std::stringstream error_message;
      error_message << "error: allocate: dimensions are " << M << 'x' << N;
      throw std::invalid_argument(error_message.str());
    }

    _data = new T[M*N];

    // validate memory allocation
    if (_data == nullptr)
    {
      throw std::bad_alloc();
    }

    // save size information
    _size      = Size{M, N, M * N};
    _true_size = _size;
  }

  template<typename T>
  void Matrix<T>::_drop(const size_t amount_) noexcept
  {
    if (amount_ > _size.numel)
    {
      PINAKAS_ERROR("invalid drop amount");
      return;
    }

    if (_size.M == 1)
    {
      _size = Size{1, N, N - amount_};
    }
    else if (_size.N == 1)
    {
      _size = Size{M, 1, M - amount_};
    }
    else
    {
      _size = Size{1, M*N - amount_, M*N - amount_};
    }

    return;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  auto Matrix<T>::operator[](const size_t j) noexcept -> T*
  {
    return _data + (j * _size.N);
  }

  template<typename T>
  auto Matrix<T>::operator[](const size_t j) const noexcept -> const T*
  {
    return _data + (j * _size.N);
  }

  template<typename T>
  auto Matrix<T>::operator[](const Indices indices_) noexcept -> T&
  {
    return (_data + (indices_.j * _size.N))[indices_.i];
  }

  template<typename T>
  auto Matrix<T>::operator[](const Indices indices_) const noexcept -> const T&
  {
    return (_data + (indices_.j * _size.N))[indices_.i];
  }

  template<typename T>
  T& Matrix<T>::operator()(signed int k) noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((k < -signed(_size.numel)) || (signed(_size.numel) <= k))
    {
      auto k_old = k;
      k %= signed(_size.numel);
      PINAKAS_WARNING("(%d) out of bound %u, wrapped around to (%d)", k_old, _size.numel, k);
    }
#endif

    // convert negative indices
    k += (k < 0) * _size.numel;

    return _data[k];
  }

  template<typename T>
  const T& Matrix<T>::operator()(signed int k) const noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((k < -signed(_size.numel)) || (signed(_size.numel) <= k))
    {
      auto k_old = k;
      k %= signed(_size.numel);
      PINAKAS_WARNING("(%d) out of bound %u, wrapped around to (%d)", k_old, _size.numel, k);
    }
#endif

    // convert negative indices
    k += (k < 0) * _size.numel;

    return _data[k];
  }

  template<typename T>
  T& Matrix<T>::operator()(signed int j, signed int i) noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((j < -signed(_size.M)) || (signed(_size.M) <= j))
    {
      auto j_old = j;
      j %= signed(_size.M);
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", j_old, _size.M, j);
    }

    // positive and negative bound checking
    if ((i < -signed(_size.N)) || (signed(_size.N) <= i))
    {
      auto i_old = i;
      i %= signed(_size.N);
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", i_old, _size.N, i);
    }
#endif

    // convert negative indices
    j += (j < 0) * _size.M;
    i += (i < 0) * _size.N;

    return _data[i + j * _size.N];
  }

  template<typename T>
  const T& Matrix<T>::operator()(signed int j, signed int i) const noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((j < -signed(_size.M)) || (signed(_size.M) <= j))
    {
      auto j_old = j;
      j %= signed(_size.M);
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", j_old, _size.M, j);
    }

    // positive and negative bound checking
    if ((i < -signed(_size.N)) || (signed(_size.N) <= i))
    {
      auto i_old = i;
      i %= signed(_size.N);
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", i_old, _size.N, i);
    }
#endif

    // convert negative indices
    j += (j < 0) * _size.M;
    i += (i < 0) * _size.N;

    return _data[i + j * _size.N];
  }
  
  template<typename T>
  Slice<T> Matrix<T>::operator()(Range rows, Range cols) noexcept
  {
    if ((rows._step != 1) || (cols._step != 1))
    {
      rows._step = 1;
      cols._step = 1;
      PINAKAS_WARNING("Range step not equal to 1 not implement yet, step = 1 used instead");
    }

# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((rows._start < -signed(_size.M)) || (signed(_size.M) <= rows._start))
    {
      auto rows_start_old = rows._start;
      rows._start %= signed(_size.M);
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", rows_start_old, _size.M, rows._start);
    }

    // positive and negative bound checking
    if ((rows._stop < -signed(_size.M)) || (signed(_size.M) <= rows._stop))
    {
      auto rows_stop_old = rows._stop;
      rows._stop %= signed(_size.M);
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", rows_stop_old, _size.M, rows._stop);
    }

    // positive and negative bound checking
    if ((cols._start < -signed(_size.N)) || (signed(_size.N) <= cols._start))
    {
      auto cols_start_old = cols._start;
      cols._start %= signed(_size.N);
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", cols_start_old, _size.N, cols._start);
    }

    // positive and negative bound checking
    if ((cols._stop < -signed(_size.N)) || (signed(_size.N) <= cols._stop))
    {
      auto cols_stop_old = cols._stop;
      cols._stop %= signed(_size.N);
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", cols_stop_old, _size.N, cols._stop);
    }
#endif

    // convert negative indices
    rows._start += (rows._start < 0) * _size.M;
    rows._stop  += (rows._stop < 0)  * _size.M;
    cols._start += (cols._start < 0) * _size.N;
    cols._stop  += (cols._stop < 0)  * _size.N;

    return Slice<T>(_data, _size, rows, cols);
  }

  template<typename T>
  Slice<const T> Matrix<T>::operator()(Range rows, Range cols) const noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((rows._start < -signed(_size.M)) || (signed(_size.M) <= rows._start))
    {
      auto rows_start_old = rows._start;
      rows._start %= signed(_size.M);
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", rows_start_old, _size.M, rows._start);
    }

    // positive and negative bound checking
    if ((rows._stop < -signed(_size.M)) || (signed(_size.M) <= rows._stop))
    {
      auto rows_stop_old = rows._stop;
      rows._stop %= signed(_size.M);
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", rows_stop_old, _size.M, rows._stop);
    }

    // positive and negative bound checking
    if ((cols._start < -signed(_size.N)) || (signed(_size.N) <= cols._start))
    {
      auto cols_start_old = cols._start;
      cols._start %= signed(_size.N);
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", cols_start_old, _size.N, cols._start);
    }

    // positive and negative bound checking
    if ((cols._stop < -signed(_size.N)) || (signed(_size.N) <= cols._stop))
    {
      auto cols_stop_old = cols._stop;
      cols._stop %= signed(_size.N);
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", cols_stop_old, _size.N, cols._stop);
    }
#endif

    // convert negative indices
    rows._start += (rows._start < 0) * _size.M;
    rows._stop  += (rows._stop < 0)  * _size.M;
    cols._start += (cols._start < 0) * _size.N;
    cols._stop  += (cols._stop < 0)  * _size.N;

    return Slice<const T>(_data, _size, rows, cols);
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
  {
    PINAKAS_LOG("copy assigned");

    // validate both matrices are not the same
    if (this != &other)
    {
      // allocate memory if necessary
      if ((_size != other._size) ||!_data)
        _allocate(other._size.M, other._size.N);

      // store values
      for (size_t k = 0; k < _size.numel; ++k)
        _data[k] = other.data()[k];
    }
    else PINAKAS_ERROR("could not copy assign, self reference is not supported");

    return *this;
  }
  template<typename T>
  Matrix<T>& Matrix<T>::operator=(const Slice<T>& other)
  {
    PINAKAS_LOG("copy assigned");

    // validate both matrices are not the same
    if (_data != other.matrix_data_)
    {
      // allocate memory if necessary
      if ((_size != other.size_) || (_data == nullptr))
      {
        _allocate(other.size_.M, other.size_.N);
      }

      // store value
      for (size_t m = 0; m < _size.M; ++m)
      {
        for (size_t n = 0; n < _size.N; ++n)
        {
          *this[m][n] = other[m][n];
        }
      }
    }
    else PINAKAS_ERROR("could not copy assign, self reference is not supported");
    
    return *this;
  }

  template<typename T>
  Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept
  {
    if (other._data != nullptr)
    {
      // take over ressources from other matrix
      _size      = other._size;
      _true_size = _size;
      _data      = other._data;

      other._size      = Size{0, 0, 0};
      other._true_size = other._size;
      other._data      = nullptr;
      
      PINAKAS_LOG("move assigned %ux%u", _size.M, _size.N);
    }
    else PINAKAS_LOG("nothing to move");

    return *this;
  }

  template<typename T>
  Matrix<T>& Matrix<T>::operator=(const T value) noexcept
  {
    PINAKAS_LOG("filled");

    // store values
    for (size_t k = 0; k < _size.numel; ++k)
      _data[k] = value;

    return *this;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Slice<T>::Slice(T* matrix_data, Size matrix_size, Range rows, Range cols) noexcept :
    matrix_data_(matrix_data),
    matrix_M_(matrix_size.M),
    offset_(rows._start * matrix_size.N + cols._start),
    size_{size_t(rows._stop - rows._start + 1)
        , size_t(cols._stop - cols._start + 1)
        , size_t((rows._stop - rows._start + 1) * (cols._stop - cols._start + 1))}
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
    
    for (size_t j = 0; j < size_.M; ++j)
    {
      for (size_t i = 0; i < size_.N; ++i)
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
      PINAKAS_ERROR("incompatible sizes (%ux%u vs %ux%u)", size_.M, size_.N, other._size.M, other._size.N);
      return *this;
    }
    
    for (size_t j = 0; j < size_.M; ++j)
    {
      for (size_t i = 0; i < size_.N; ++i)
      {
        matrix_data_[i + j*matrix_M_ + offset_] = other[j][i];
      }
    }

    return *this;
  }
  
  template<typename T>
  Slice<T>& Slice<T>::operator=(T value)
  {
    for (size_t j = 0; j < size_.M; ++j)
    {
      for (size_t i = 0; i < size_.N; ++i)
      {
        matrix_data_[i + j*matrix_M_ + offset_] = value;
      }
    }

    return *this;
  }

  template<typename T>
  T& Slice<T>::operator()(signed int k) noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((k < -signed(size_.numel)) || (signed(size_.numel) <= k))
    {
      auto k_old = k;
      k %= signed(size_.numel);
      PINAKAS_WARNING("(%d) out of bound %u, wrapped around to (%d)", k_old, size_.numel, k);
    }
#endif

    // convert negative indices
    k += (k < 0) * size_.numel;

    // compute 2D indices
    const size_t i = k % size_.N;
    const size_t j = k / size_.N;

    return matrix_data_[i + j*matrix_M_ + offset_];
  }

  template<typename T>
  const T& Slice<T>::operator()(signed int k) const noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((k < -signed(size_.numel)) || (signed(size_.numel) <= k))
    {
      auto k_old = k;
      k %= signed(size_.numel);
      PINAKAS_WARNING("(%d) out of bound %u, wrapped around to (%d)", k_old, size_.numel, k);
    }
#endif

    // convert negative indices
    k += (k < 0) * size_.numel;

    // compute 2D indices
    size_t i = k % size_.N;
    size_t j = k / size_.N;

    return matrix_data_[i + j*matrix_M_ + offset_];
  }

  template<typename T>
  T& Slice<T>::operator()(signed int j, signed int i) noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((j < -signed(size_.M)) || (signed(size_.M) <= j))
    {
      auto j_old = j;
      j %= signed(size_.M);
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", j_old, size_.M, j);
    }

    // positive and negative bound checking
    if ((i < -signed(size_.N)) || (signed(size_.N) <= i))
    {
      auto i_old = i;
      i %= signed(size_.N);
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", i_old, size_.N, i);
    }
#endif

    // convert negative indices
    j += (j < 0) * size_.M;
    i += (i < 0) * size_.N;

    return matrix_data_[i + j*matrix_M_ + offset_];
  }

  template<typename T>
  const T& Slice<T>::operator()(signed int j, signed int i) const noexcept
  {
# if defined PINAKAS_DEBUG_MODE
    // positive and negative bound checking
    if ((j < -signed(size_.M)) || (signed(size_.M) <= j))
    {
      auto j_old = j;
      j %= signed(size_.M);
      PINAKAS_WARNING("(%d, _) out of bound %u, wrapped around to (%d, _)", j_old, size_.M, j);
    }

    // positive and negative bound checking
    if ((i < -signed(size_.N)) || (signed(size_.N) <= i))
    {
      auto i_old = i;
      i %= signed(size_.N);
      PINAKAS_WARNING("(_, %d) out of bound %u, wrapped around to (_, %d)", i_old, size_.N, i);
    }
#endif

    // convert negative indices
    j += (j < 0) * size_.M;
    i += (i < 0) * size_.N;

    return matrix_data_[i + j*matrix_M_ + offset_];
  }
// --------------------------------------------------------------------------------------
  Random::Random(const double min, const double max) noexcept :
    min_(min),
    max_(max)
  {}
// --------------------------------------------------------------------------------------
  Range::Range(const size_t high_) noexcept :
    _start(0),
    _stop(signed(high_ - 1)),
    _step(1)
  {}

  Range::Range(const int start_, const int stop_) noexcept :
    _start(start_),
    _stop(stop_),
    _step(((_stop-_start) >= 0) ? 1 : -1)
  {}

  Range::Range(const int start_, const int stop_, const size_t step_) noexcept :
    _start(start_),
    _stop(stop_),
    _step(((_stop-_start) >= 0) ? signed(step_) : -signed(_step))
  {}

  Set::Set(const char* name_, const Matrix<double>& x_, const Matrix<double>& y_) noexcept :
    name(name_),
    x(x_),
    y(y_)
  {}

  Set::Set(const char* name_, const Matrix<double>& y_) noexcept :
    _x_if_temp(linspace(0, 1, y_.numel())),
    name(name_),
    x(_x_if_temp),
    y(y_)
  {}
//----------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& _div_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return A;
    }

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A.data()[k] /= B.data()[k];

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& _div_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return B;
    }

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B.data()[k] = A.data()[k] / B.data()[k];

    return B;
  }

  template<typename T1, typename T2>
  Matrix<T1>& _div_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const T1 iB = 1.0 / B;

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A.data()[k] *= iB;

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& _div_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept
  {
    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B.data()[k] = A / B.data()[k];

    return B;
  }

  template<typename T>
  Matrix<T>& _div_ll_rng_inplace(Matrix<T>& A, Random B) noexcept
  {
    static std::random_device device;
    static std::mt19937 generator(device());
    std::uniform_real_distribution<float> uniform_distribution(B.min_, B.max_ + std::is_integral<T>::value);
    
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A.data()[k] /= uniform_distribution(generator);

    return A;
  }

  template<typename T>
  Matrix<T>& _div_rl_rng_inplace(Matrix<T>& B, Random A) noexcept
  {
    static std::random_device device;
    static std::mt19937 generator(device());
    std::uniform_real_distribution<float> uniform_distribution(A.min_, A.max_ + std::is_integral<T>::value);

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B.data()[k] = uniform_distribution(generator) / B.data()[k];

    return B;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> _div_mat(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size())
    {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return Matrix<T3>();
    }

    Matrix<T3> result(A.size());

    const T1* a = A.data();
    const T2* b = B.data();
    T3* r = result.data();

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      r[k] = a[k] / b[k];
    
    return result;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> _div_ll_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> result(A.size());
    const T1 iB = 1.0 / B;

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = A.data()[k] * iB;

    return result;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> _div_rl_val(const Matrix<T1>& B, const T2 A) noexcept
  {
    Matrix<T3> result(B.size());

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = A / B.data()[k];

    return result;
  }

  template<typename T1, typename T3>
  Matrix<T3> _div_ll_rng(const Matrix<T1>& A, Random B) noexcept
  {
    static std::random_device device;
    static std::mt19937 generator(device());
    std::uniform_real_distribution<float> uniform_distribution(B.min_, B.max_ + std::is_integral<T1>::value);

    Matrix<T3> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = A.data()[k] / uniform_distribution(generator);

    return result;
  }

  template<typename T1, typename T3>
  Matrix<T3> _div_rl_rng(const Matrix<T1>& B, Random A) noexcept
  {
    static std::random_device device;
    static std::mt19937 generator(device());
    std::uniform_real_distribution<float> uniform_distribution(A.min_, A.max_ + std::is_integral<T1>::value);

    Matrix<T3> result(B.size());

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = uniform_distribution(generator) / B.data()[k];

    return result;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator/=(Matrix<T1>& A, const Matrix<T2>& B)
  {
    return _div_ll_mat_inplace(A, B);
  }
  
  template<typename T>
  Matrix<T>& operator/=(Matrix<T>& A, Random B) noexcept
  {
    return _div_ll_rng_inplace(A, B);
  }

  template<typename T1, typename T2>
  Matrix<T1>& operator/=(Matrix<T1>& A, const T2 B) noexcept
  {
    return _div_ll_val_inplace(A, B);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator/(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return _div_mat(A, B);
  }
  
  template<typename T>
  Matrix<T> operator/(const Matrix<T>& A, const Matrix<T>& B)
  {
    #ifdef PARALLILOS_USE_PARALLELISM
      // return div_mat_simd(A, B);
    #else
      return _div_mat(A, B);
    #endif
  }

  template<typename T, typename T3>
  Matrix<T3> operator/(const Matrix<T>& A, Random B) noexcept
  {
    return _div_ll_rng(A, B);
  }

  template<typename T, typename T3>
  Matrix<T3> operator/(Random A, const Matrix<T>& B) noexcept
  {
    return _div_rl_rng(B, A);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator/(const Matrix<T1>& A, const T2 B) noexcept
  {
    return _div_ll_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator/(const T1 A, const Matrix<T2>& B) noexcept
  {
    return _div_rl_val(B, A);
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator/(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(_div_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(_div_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&
  {
    return std::move(_div_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&
  {
    return std::move(_div_ll_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator/(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(_div_ll_mat_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator/(Matrix<T>&& A, Random B) noexcept
  {
    return std::move(_div_ll_rng_inplace(A, B));
  }

  template<typename T, typename T3>
  Matrix<T3>&& operator/(Random A, Matrix<T>&& B) noexcept
  {
    return std::move(_div_rl_rng_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator/(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(_div_rl_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(_div_ll_val_inplace(A, B));
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& _pow_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return A;
    }

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A.data()[k] = std::pow(A.data()[k], B.data()[k]);

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& _pow_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return B;
    }

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B.data()[k] = std::pow(A.data()[k], B.data()[k]);

    return B;
  }

  template<typename T1, typename T2>
  Matrix<T1>& _pow_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A.data()[k] = std::pow(A.data()[k], B);

    return A;
  }

  template<typename T1, typename T2>
  Matrix<T1>& _pow_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept
  {
    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      B.data()[k] = std::pow(A, B.data()[k]);

    return B;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> _pow_mat(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    if (A.size() != B.size()) {
      PINAKAS_ERROR("nonconformant arguments (A is %ux%u, B is %ux%u)", A.M(), A.N(), B.M(), B.N());
      return Matrix<T3>();
    }

    Matrix<T3> result(A.size());
  
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::pow(A.data()[k], B.data()[k]);

    return result;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> _pow_ll_val(const Matrix<T1>& A, const T2 B) noexcept
  {
    Matrix<T3> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::pow(A.data()[k], B);

    return result;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> _pow_rl_val(const Matrix<T1>& B, const T2 A) noexcept
  {
    Matrix<T3> result(B.size());

    const size_t n = B.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::pow(A, B.data()[k]);

    return result;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator^=(Matrix<T1>& A, const Matrix<T2>& B)
  {
    return _pow_ll_mat_inplace(A, B);
  }

  template<typename T1, typename T2>
  Matrix<T1>& operator^=(Matrix<T1>& A, const T2 B) noexcept
  {
    return _pow_ll_val_inplace(A, B);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator^(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    return _pow_mat(A, B);
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator^(const Matrix<T1>& A, const T2 B) noexcept
  {
    return _pow_ll_val(A, B);
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3> operator^(const T1 A, const Matrix<T2>& B) noexcept
  {
    return _pow_rl_val(B, A);
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator^(const Matrix<T1>& A, Matrix<T2>&& B)
  {
    return std::move(_pow_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const Matrix<T2>& B)
  {
    return std::move(_pow_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&
  {
    return std::move(_pow_rl_mat_inplace(B, A));
  }
  
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&
  {
    return std::move(_pow_ll_mat_inplace(A, B));
  }
  
  template<typename T>
  Matrix<T>&& operator^(Matrix<T>&& A, Matrix<T>&& B)
  {
    return std::move(_pow_ll_mat_inplace(A, B));
  }
  
  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator^(const T1 A, Matrix<T2>&& B) noexcept
  {
    return std::move(_pow_rl_val_inplace(B, A));
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const T2 B) noexcept
  {
    return std::move(_pow_ll_val_inplace(A, B));
  }
// --------------------------------------------------------------------------------------
  template<template<typename> class M, typename T>
  auto floor(const M<T>& A) -> Matrix<T>
  {
    static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
    Matrix<T> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
    {
      result.data()[k] = std::floor(A.data()[k]);
    }

    return result;
  }

  template<template<typename> class M, typename T>
  auto floor(M<T>&& A) noexcept -> M<T>&&
  {
    static_assert(std::is_floating_point<T>::value, "floor: T must be a floating-point type");
    static_assert(not std::is_const<T>::value, "floor: T must not be const");

    for (T& data : A)
    {
      data = std::floor(data);
    }

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<template<typename> class M, typename T>
  auto round(const M<T>& A) -> Matrix<T>
  {
    static_assert(std::is_floating_point<T>::value, "round: T must be a floating-point type");
    Matrix<T> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
    {
      result.data()[k] = std::round(A.data()[k]);
    }

    return result;
  }

  template<template<typename> class M, typename T>
  auto round(M<T>&& A) noexcept -> M<T>&&
  {
    static_assert(std::is_floating_point<T>::value, "round: T must be a floating-point type");
    static_assert(not std::is_const<T>::value, "round: T must not be const");

    for (T& data : A)
    {
      data = std::round(data);
    }

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<template<typename> class M, typename T>
  auto ceil(const M<T>& A) -> Matrix<T>
  {
    static_assert(std::is_floating_point<T>::value, "ceil: T must be a floating-point type");
    Matrix<T> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
    {
      result.data()[k] = std::ceil(A.data()[k]);
    }

    return result;
  }

  template<template<typename> class M, typename T>
  auto ceil(M<T>&& A) noexcept -> M<T>&&
  {
    static_assert(std::is_floating_point<T>::value, "ceil: T must be a floating-point type");
    static_assert(not std::is_const<T>::value, "ceil: T must not be const");

    for (T& data : A)
    {
      data = std::ceil(data);
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

    for (size_t i = 0; i < B.N(); i++)
    {
      for (size_t j = 0; j < A.M(); j++)
      {
        for (size_t k = 0; k < A.N(); k++)
        {
          result[j][i] += A[j][k] * B[k][i];
        }
      }
    }

    return result;
  }

  template<typename T>
  Matrix<T> div(const Matrix<T>& b, Matrix<T> A)
  {
    Matrix<double> x;

    // verify vertical dimensions
    if (b.M() != A.M()) 
    {
      PINAKAS_ERROR("vertical dimensions mismatch (b is %zux_, A is %zux_)", b.M(), A.M());
      return x;
    }

    // verify that b is a column matrix
    if (b.N() != 1)
    {
      PINAKAS_ERROR("b's horizontal dimension is not 1 (b is _x%zu", b.N());
      return x;
    }

    // store the dimensions of A
    const size_t M = A.M();
    const size_t N = A.N();

    // QR decomposition matrices and result matrix
    Matrix<double> Q(M, N), R(N, N);
    x = Matrix<double>(N, 1);

    // QR decomposition using the modified Gram-Schmidt process
    for (size_t i = 0; i < N; ++i)
    {
      // calculate the squared Euclidean norm of A's i'th column
      double sum_of_squares = 0;
      for (size_t j = 0; j < M; ++j)
      {
        sum_of_squares += A[j][i] * A[j][i];
      }

      // skips if the squared Euclidean norm is 0
      if (sum_of_squares != 0)
      {
        // calculate the inverse Euclidean norm of A's i'th column
        double inorm = std::pow(sum_of_squares, -0.5);
        // normalize and store A's normalized i'th column
        for (size_t j = 0; j < M; ++j)
        {
          Q[j][i] = A[j][i] * inorm;
        }
      }

      // orthogonalize the remaining columns with respects to A's i'th column
      for (size_t k = i; k < N; ++k)
      {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        double projection = 0;
        for (size_t j = 0; j < M; ++j)
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
          for (size_t j = 0; j < M; ++j)
          {
            A[j][k] -= projection * Q[j][i];
          }
        }
      }
    }

    // solve linear system Rx = Qt*b using back substitution
    for (size_t i = N - 1; i < N; --i)
    {
      // calculate appropriate Qt*b component
      double substitution = 0;
      for (size_t j = 0; j < M; ++j)
      {
        substitution += Q[j][i] * b[j][0];
      }

      // back substitution of previously solved x components
      for (size_t k = N - 1; k > i; --k)
      {
        substitution -= R[i][k] * x[k][0];
      }

      // solve x's i'th component
      x[i][0] = substitution / R[i][i];
    }

    return x;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  auto transpose(const Matrix<T>& A) -> Matrix<T>
  {
    Matrix<T> result(A.N(), A.M());
    for (size_t y = 0; y < A.M(); ++y)
      for (size_t x = 0; x < A.N(); ++x)
        result[x][y] = A[y][x];
    return result;
  }

  template<typename T>
  auto transpose(Matrix<T>&& A) noexcept -> Matrix<T>&&
  {
    std::swap(A._size.M, A._size.N);

    return std::move(A);
  }

  template<typename T>
  auto reshape(const Matrix<T>& A, size_t M, size_t N) -> Matrix<T>
  {
    Matrix<T> result;

    if (A.numel() != M * N)
    {
      PINAKAS_ERROR("can't reshape %ux%u into %ux%u)", A.M(), A.N(), M, N);
      return result;
    }

    result = Matrix<T>(M, N);

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
    {
      result.data()[k] = A.data()[k];
    }

    return result;
  }

  template<typename T>
  auto reshape(Matrix<T>&& A, size_t M, size_t N) noexcept -> Matrix<T>&&
  {
    if (A._size.numel != M*N)
    {
      PINAKAS_ERROR("can't reshape %ux%u into %ux%u)", A.M(), A.N(), M, N);
      return std::move(A);
    }

    A._size.M = M;
    A._size.N = N;

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<template<typename> class M, typename T>
  T min(const M<T>& A) noexcept
  {
    T result = *A.begin();

    for (T data : A)
    {
      if (data < result)
      {
        result = data;
      }
    }
        
    return result;
  }
  
  template<template<typename> class M, typename T>
  T max(const M<T>& A) noexcept
  {
    T result = *A.begin();

    for (T data : A)
    {
      if (data > result)
      {
        result = data;
      }
    }

    return result;
  }
  
  template<template<typename> class M, typename T>
  T sum(const M<T>& A) noexcept
  {
    T result = 0;

    for (T data : A)
    {
      result += data;
    }

    return result;
  }

  template<template<typename> class M, typename T>
  double prod(const M<T>& A) noexcept
  {
    // double result = 0;
    // for (T data : A)
    // {
    //   result += std::log(data);
    // }
    // return std::exp(result);

    double result = 1.0;

    for (T data : A)
    {
      result *= data;
    }

    return result;
  }

  template<template<typename> class M1, typename T>
  double avg(const M1<T>& A) noexcept
  {
    double average = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      average += A.data()[k];

    return average/A.numel();
  }

  template<template<typename> class M1, typename T>
  double rms(const M1<T>& A) noexcept
  {
    const size_t n = A.numel();
    double temporary = 0;
    for (size_t k = 0; k < n; ++k)
      temporary += A.data()[k] * A.data()[k];
    
    return std::sqrt(temporary);
  }

  template<template<typename> class M1, typename T>
  double geo(const M1<T>& A) noexcept
  {
    double temporary = 0;
    for (size_t k = 0; k < A.numel(); ++k)
      temporary += std::log(A.data()[k]);

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
      for (size_t k = i; k < N; ++k)
      {
        // calculate Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        double projection = 0;
        for (size_t j = 0; j < M; ++j)
          projection += Q[j][i] * A[j][k];

        // orthogonalize A's k'th column by removing Q's i'th orthonormal projection onto A's k'th unorthogonalized column
        if (k != i) // skips if k == i because the projection would be 0
        {
          for (size_t j = 0; j < M; ++j)
          {
            A[j][k] -= projection * Q[j][i];
          }
        }
      }
    }

    return Q;
  }

  std::unique_ptr<Matrix<double>[]> qr(Matrix<double> A)
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
    auto result = std::unique_ptr<Matrix<double>[]>(new Matrix<double>[2]);
    if (data_x.numel() != data_y.numel())
    {
      PINAKAS_ERROR("'data_x' and 'data_y' must have the same number of elements");
      return result;
    }

    PINAKAS_WARNING_IF((data_x.M() != 1) || (data_y.M() != 1), "data is interpreted as a horizontal 1-dimensional matrix");

    const size_t n = data_x.numel();

    Matrix<double> lin_x(1, n, 0), lin_y(1, n, 0);

    // set first and last values of the linearized data set
    lin_x.data()[0]   = data_x.data()[0];
    lin_y.data()[0]   = data_y.data()[0];
    lin_x.data()[n-1] = data_x.data()[n-1];
    lin_y.data()[n-1] = data_y.data()[n-1];

    // build linearly spaced x data and its associated y value
    const double step = (data_x.data()[n-1] - data_x.data()[0]) / static_cast<double>(n - 1);
    for (size_t k = 1; k < (n - 1); ++k)
    {
      // build linearly spaced x data
      lin_x.data()[k] = lin_x.data()[k-1] + step;

      // linearly interpolate y value
      double x1 = data_x.data()[k];
      double y1 = data_y.data()[k];
      double x2 = data_x.data()[k+1];
      double y2 = data_y.data()[k+1];
      lin_y.data()[k] = y1 + (lin_x.data()[k] - x1) * (y2 - y1) / (x2 - x1);
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

    PINAKAS_WARNING_IF((data_x.M() != 1) || (data_y.M() != 1), "data is interpreted as a horizontal 1-dimensional matrix");

    const size_t n = data_x.numel();

    // build linearly spaced x data and its associated y value
    const double step = (data_x.data()[n - 1] - data_x.data()[0]) / (n - 1);
    for (size_t k = 1; k < (n - 1); ++k)
    {
      // store necessary data to linearly interpolate y value
      double x1 = data_x.data()[k];
      double y1 = data_y.data()[k];
      double x2 = data_x.data()[k+1];
      double y2 = data_y.data()[k+1];

      // build linearly spaced data
      data_x.data()[k] = data_x.data()[k - 1] + step;
      data_y.data()[k] = y1 + (data_x.data()[k] - x1) * (y2 - y1) / (x2 - x1);
    }

    result[0] = std::move(data_x);
    result[1] = std::move(data_y);
    return result;
  }

  template<typename T>
  Matrix<T> linspace(const double x1, const double x2, const size_t N)
  {
    Matrix<T> result(1, N);

    const auto nm1   = N - 1;
    const auto step  = (x2 - x1) / static_cast<double>(nm1);
    auto temporary   = x1;

    if (std::is_floating_point<T>::value)
    {
      for (size_t k = 1; k < nm1; ++k)
      {
        temporary       += step;
        result.data()[k] = temporary;
      }
    }
    else
    {
      for (size_t k = 1; k < nm1; ++k)
      {
        temporary       += step;
        result.data()[k] = std::round(temporary);
      }
    }

    result.data()[0]   = x1;
    result.data()[nm1] = x2;

    return result;
  }

  template<typename T>
  auto iota(const size_t N) -> Matrix<T>
  {
    auto result = Matrix<T>(1, N);

    size_t value = 0;
    for (auto& data : result.data())
    {
      data = value++;
    }

    return result;
  }

  template<typename T>
  auto eye(const size_t M, const size_t N) -> Matrix<T>
  {
    auto result = Matrix<T>(M, N, 0);

    const auto N_ = std::min(M, N);
    for (size_t k = 0; k < N_; ++k)
    {
      result[k][k] = 1;
    }
    
    return result;
  }

  Matrix<double> diff(const Matrix<double>& A, size_t n)
  {
    if (n)
    {
      Matrix<double> result(A.M(), A.N() - 1);
      for (size_t y = 0; y < result.M(); ++y)
      {
        for (size_t x = 0; x < result.N(); ++x)
        {
          result[y][x] = A[y][x + 1] - A[y][x];
        }
      }

      return diff(result, n - 1);
    }

    return A;
  }

  template<typename T>
  Matrix<T> reverse(const Matrix<T>& A)
  {
    auto result = Matrix<T>(A.M(), A.N());

    const auto N = A.numel();
    for (size_t k = 0; k < N; ++k)
    {
      result.data()[k] = A.data()[N-1 - k];
    }

    return result;
  }

  template<typename T>
  Matrix<T> reverse(const Slice<T>& A)
  {
    Matrix<T> result(A.size());

    const size_t N = A.numel();
    for (size_t k = 0; k < N; ++k)
    {
      result.data()[k] = A[0][N-1 - k];
    }

    return result;
  }

  template<template<typename> class M, typename T>
  M<T>&& reverse(M<T>&& A) noexcept
  {
    static_assert(not std::is_const<T>::value, "reverse: T must not be const");

    const size_t n   = A.numel();
    const size_t n_2 = n >> 1;
    for (size_t k = 0; k < n_2; ++k)
    {
      std::swap(A.data()[k], A.data()[n-1 - k]);
    }

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  auto conv(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()*T2())>
  {
    Matrix<decltype(T1()*T2())> result;

    if (A.M() != B.M())
    {
      PINAKAS_ERROR("vertical dimensions mismatch (%u vs %u)", A.M(), B.M());
      return result;
    }

    const size_t M   = A.M();
    const size_t N_A = A.N();
    const size_t N_B = B.N();

    result = Matrix<decltype(T1()*T2())>(1, N_A+N_B-1, 0);

    for (size_t m = 0; m < M; ++m)
    {
      for (size_t i = 0; i < N_A; ++i)
      {
        for (size_t j = 0; j < N_B; ++j)
        {
          result[m][i + j] += A[m][i] * B[m][j];
        }
      }
    }

    return result;
  }

  template<typename T1, typename T2, typename T3>
  Matrix<T3> corr(const Matrix<T1>& A, const Matrix<T2>& B)
  {
    const size_t n1 = A.numel();
    const size_t n2 = B.numel();

    Matrix<T3> result(1, n1 + n2 - 1, 0);
    for (size_t i = 0; i < n1; ++i)
      for (size_t j = 0; j < n2; ++j)
        result.data()[i + j] += A.data()[n1-1 - i] * B.data()[j];

    return result; 
  }

  template<typename T>
  Matrix<T> corr(const Matrix<T>& A)
  {
    const size_t n = A.numel();

    Matrix<T> result(1, 2*n - 1, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        result.data()[i + j] += A.data()[n-1 - i] * A.data()[j];

    return result;
  }

  Matrix<double> Rxx(const Matrix<double>& A)
  {
    const size_t n = A.numel();

    Matrix<double> result(1, n, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if ((i+j - n+1) < n)
          result.data()[i+j - n+1] += A.data()[n-1 - i] * A.data()[j];

    return result;
  }

  Matrix<double> Rxx(const Matrix<double>& A, const size_t K)
  {
    const size_t n = A.numel();

    Matrix<double> result(1, K, 0);
    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < n; ++j)
      {
        if ((i+j - n+1) < K)
        {
          result.data()[i+j - n+1] += A.data()[n-1 - i] * A.data()[j];
        }
      }
    }

    return result;
  }

  Matrix<double> lpc(const Matrix<double>& A, const size_t p)
  {
    auto result = Matrix<double>();

    if (p >= A.numel())
    {
      PINAKAS_ERROR("'p' should be less than number of elements in 'A'");
      return result;
    }

    PINAKAS_WARNING_IF(A.M() != 1, "'A' should be a horizontal vector");

    const auto rxx      = Rxx(A, p+1);
    const auto rxx_data = rxx.data();

    auto autocorr_mat = Matrix<double>(p, p);
    auto autocorr_vec = Matrix<double>(p, 1);
    auto vec_data     = autocorr_vec.data();

    for (size_t i = 0; i < p; ++i)
    {
      for (size_t j = 0; j < p; ++j)
      {
        autocorr_mat[j][i] = rxx_data[(j > i) ? (j - i) : (i - j)];
      }

      vec_data[i] = rxx_data[i+1];
    }

    // TO DO: Levinson recursion (Levinson-Durbin algorithm)
    // to solve Toeplitz system of equation more efficiently
    result = div(autocorr_vec, autocorr_mat);

    return result;
  }

  Matrix<double> toeplitz(const Matrix<double>& A)
  {
    const auto N    = A.numel();
    const auto data = A.data();

    auto result     = Matrix<double>(N, N);
    for (size_t i = 0; i < N; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        result[j][i] = data[(j > i) ? (j - i) : (i - j)];
      }
    }

    return result;
  }
// --------------------------------------------------------------------------------------
  Matrix<double> blackman(size_t N)
  {
    Matrix<double> window(1, N);
    for (size_t k = 0; k < N; ++k)
    {
      const double temporary = (2 * M_PI * k)/static_cast<double>(N - 1);
      window.data()[k] = 0.42 - 0.50 * std::cos(temporary) + 0.08 * std::cos(2 * temporary);
    }

    return window;
  }

  Matrix<double> blackman(const Matrix<double>& signal)
  {
    const size_t N = signal.numel();
    Matrix<double> windowed(1, N);
    for (size_t k = 0; k < N; ++k)
    {
      const double temporary = (2 * M_PI * k)/static_cast<double>(N - 1);
      windowed.data()[k] = signal.data()[k] * (0.42 - 0.50 * std::cos(temporary) + 0.08 * std::cos(2 * temporary));
    }

    return windowed;
  }

  Matrix<double>&& blackman(Matrix<double>&& signal) noexcept
  {
    const auto N = signal.numel();
    auto data    = signal.data();
    for (size_t k = 0; k < N; ++k)
    {
      const double temporary = (2 * M_PI * k)/static_cast<double>(N - 1);
      data[k] *= 0.42 - 0.50 * std::cos(temporary) + 0.08 * std::cos(2 * temporary);
    }

    return std::move(signal);
  }
// --------------------------------------------------------------------------------------
  Matrix<double> hamming(size_t N)
  {
    Matrix<double> window(1, N);

    for (size_t k = 0; k < N; ++k)
    {
      window.data()[k] = 0.54 - 0.46 * std::cos(2 * M_PI * k / static_cast<double>(N - 1));
    }

    return window;
  }

  Matrix<double> hamming(const Matrix<double>& signal)
  {
    const auto N = signal.numel();
    Matrix<double> windowed(1, N);
    for (size_t k = 0; k < N; ++k)
    {
      windowed.data()[k] = signal.data()[k] * (0.54 - 0.46 * std::cos(2 * M_PI * k / static_cast<double>(N - 1)));
    }

    return windowed;
  }

  Matrix<double>&& hamming(Matrix<double>&& signal) noexcept
  {
    const auto N = signal.numel();
    auto data    = signal.data();
    for (size_t k = 0; k < N; ++k)
    {
      data[k] *= 0.54 - 0.46 * std::cos(2 * M_PI * k / static_cast<double>(N - 1));
    }

    return std::move(signal);
  }
// --------------------------------------------------------------------------------------
  Matrix<double> hann(size_t N)
  {
    Matrix<double> window(1, N);
    for (size_t k = 0; k < N; ++k)
    {
      window.data()[k] = 0.5 - 0.5 * std::cos(2 * M_PI * k / static_cast<double>(N - 1));
    }

    return window;
  }

  Matrix<double> hann(const Matrix<double>& signal)
  {
    const auto N = signal.numel();
    Matrix<double> windowed(1, N);
    for (size_t k = 0; k < N; ++k)
    {
      windowed.data()[k] = signal.data()[k] * (0.5 - 0.5 * std::cos(2 * M_PI * k / static_cast<double>(N - 1)));
    }

    return windowed;
  }

  Matrix<double>&& hann(Matrix<double>&& signal) noexcept
  {
    const auto N = signal.numel();
    auto data    = signal.data();
    for (size_t k = 0; k < N; ++k)
    {
      data[k] *= 0.5 - 0.5 * std::cos(2*M_PI*k / static_cast<double>(N - 1));
    }

    return std::move(signal);
  }
// --------------------------------------------------------------------------------------
  double newton(std::function<double(double)> function,
    double tol,
    size_t max_iteration,
    double seed) noexcept
  {
    const double half_tol = tol * 0.5;

    double root = seed;

    size_t iteration = 0;
    while ((tol < std::abs(function(root))) && (iteration++ < max_iteration))
    {
      root -= tol * function(root) / (function(root + half_tol) - function(root - half_tol));
    }

    return root;
  }
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<double> cos(const Matrix<T>& A)
  {
    Matrix<double> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::cos(A.data()[k]);

    return result;
  }

  Matrix<double>&& cos(Matrix<double>&& A) noexcept
  {
    for (auto& value : A)
    {
      value = std::cos(value);
    }

    return std::move(A);
  }

  template<typename T>
  Matrix<double> sin(const Matrix<T>& A)
  {
    Matrix<double> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::sin(A.data()[k]);

    return result;
  }

  Matrix<double>&& sin(Matrix<double>&& A) noexcept
  {
    for (auto& value : A)
    {
      value = std::sin(value);
    }

    return std::move(A);
  }

  template<typename T>
  Matrix<double> sinc(const Matrix<T>& A)
  {
    Matrix<double> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
    {
      double temporary = M_PI * A.data()[k];
      result.data()[k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return result;
  }

  Matrix<double>&& sinc(Matrix<double>&& A) noexcept
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
    {
      double temporary = M_PI * A.data()[k];
      A.data()[k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return std::move(A);
  }
// --------------------------------------------------------------------------------------
  Matrix<double> upsample(const Matrix<double>& data, const size_t L)
  {
    const size_t n = data.numel();
    Matrix<double> upsampled(1, L * n, 0);

    for (size_t k = 0, kL = 0; k < n; ++k, kL += L)
    {
      upsampled.data()[kL] = data.data()[k];
    }

    return upsampled;
  }

  Matrix<double> sinc_impulse(const size_t length, const double frequency)
  {
    Matrix<double> impulse;

    // validate impulse length
    if ((length % 2) == 0)
    {
      PINAKAS_ERROR("'length' must be odd");
      return impulse;
    }

    // offset to the impulse center
    const signed int offset = static_cast<double>(length - 1)/2;

    // compute impulse
    impulse = Matrix<double>(1, length);
    for (signed int k = 0; k < signed(length); ++k)
    {
      const double temporary = M_PI * (k - offset) * frequency;
      impulse.data()[k] = (temporary == 0) ? 1 : std::sin(temporary) / temporary;
    }

    return impulse;
  }

  template<typename T>
  Matrix<double> resample(const Matrix<T>& data, size_t L, size_t reflect, float alpha, const bool tail)
  {
    static_assert(std::is_convertible<T, double>::value, "T must be convertible to double");
    Matrix<double> resampled;

    if (data.N() < 4)
    {
      PINAKAS_ERROR("'data' must be atleast 4 element wide");
      return resampled;
    }

#  if defined PINAKAS_DEBUG_MODE
    if (L <= 1)
    {
      L = 2;
      PINAKAS_WARNING("'L' must be at least 2, 2 used instead");
    }
    
    if (reflect >= data.N())
    {
      reflect = 2;
      PINAKAS_WARNING("'reflect' must be less than the width of 'data', 2 used instead");
    }
    
    if (alpha < (1.0f/L))
    {
      alpha = 3.5f;
      PINAKAS_WARNING("'alpha' must be at least 1/L, 3.5 used instead");
    }
# endif

    // design low-pass interpolation filter
    const size_t offset        = static_cast<size_t>(L * alpha); // offset to impulse center
    const size_t filter_length = 2 * offset + 1;
    static auto    filter_cached = blackman(sinc_impulse(filter_length, 1.0/L));
    static auto    offset_cached = offset;
    static auto    factor_cached = L;
    if ((offset_cached != offset) || (factor_cached != L))
    {
      filter_cached = blackman(sinc_impulse(filter_length, 1.0/L));
      offset_cached = offset;
      factor_cached = L;
    }

    const size_t N = data.N();
    const size_t M = data.M();

    // indices to the first and last upsampled elements in the symetrically extended data
    const size_t first = L*reflect;
    const size_t last  = L*N + first - (tail ? 1 : L);

    // symetrically extended data
    const size_t extended_length = N + 2 * reflect;
    Matrix<double> extended(M, extended_length, 0);

    resampled = Matrix<double>(M, last - first + 1, 0);
    for (size_t m = 0; m < M; ++m) // resample every row separately
    {
      size_t k = 0;
      for (size_t i = 0; i < reflect; ++i, ++k) // store and upsample left symetrical data
      {
        extended[m][k] = 2*data[m][0] - data[m][reflect - i];
      }

      for (size_t i = 0; i < N; ++i, ++k)       // store and upsample data
      {
        extended[m][k] = data[m][i];
      }

      for (size_t i = 0; i < reflect; ++i, ++k) // store and upsample right symetrical data
      {
        extended[m][k] = 2*data[m][N-1] - data[m][N-2 - i];
      }

      // interpolate upsampled data using a cropped convolution
      for (size_t i = 0; i < extended_length; ++i)
      {
        for (size_t j = 0; j < filter_length; ++j)
        {
          k = i*L + j - offset;
          // skips if the index is not within the upsampled data range (cropping)
          if ((first <= k) && (k <= last))
          {
            resampled[m][k - first] += extended[m][i] * filter_cached[m][j];
          }
        }
      }
    }

    return resampled;
  }

  template<template<typename> class M, typename T, typename>
  std::ostream& operator<<(std::ostream& ostream, const M<T>& A)
  {
    if (A.numel() != 0)
    {
      if (A.M() != 1)
      {
        size_t max_len = 0;
        for (T data : A)
        {
          std::stringstream ss;
          ss.copyfmt(ostream);
          ss << data;
          max_len = std::max(size_t(max_len), ss.str().length());
        }
      
        for (size_t y = 0; y < A.M(); ++y)
        {
          for (size_t x = 0; x < A.N(); ++x)
          {
            ostream << std::setw(max_len + 1) << A[y][x];
          }
          ostream << '\n';
        }
      }
      else
      {
        for (T data : A)
        {
          ostream << data << ' ';
        }
      }

      ostream << '\n';
    }

    return ostream;
  }

  std::ostream& operator<<(std::ostream& ostream, const Size size)
  {
    return ostream << size.M << 'x' << size.N;
  }

  __attribute__((optimize("O0")))
  void plot(List<Set> data_sets, bool persistent, bool remove, bool pause, bool lines)
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
    for (const auto& data_set : data_sets)
    {
      if (data_set.x.numel() != data_set.y.numel())
      {
        PINAKAS_ERROR("number of element mismatch (x has %zu elements, y has %zu elements)", data_set.x.numel(), data_set.y.numel());
        return;
      }
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
      for (size_t k = 0; k < data_set.x.numel(); ++k)
      {
        file << data_set.x.data()[k] << ' ' << data_set.y.data()[k] << '\n';
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
    size_t k = 0;
    for (const auto& data_set : data_sets) {
      if (k)
        gnuplot_pipeline << ", ''";
      gnuplot_pipeline << " index " << k;
      if (lines)
        gnuplot_pipeline << " with lines";
      gnuplot_pipeline << " title '" << data_set.name << '\'';
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
    Matrix<double> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::abs(A.data()[k]);

    return result;
  }
  
  Matrix<double> real(const Matrix<complex>& A)
  {
    Matrix<double> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::real(A.data()[k]);

    return result;
  }
  
  Matrix<double> imag(const Matrix<complex>& A)
  {
    Matrix<double> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::imag(A.data()[k]);

    return result;
  }

  Matrix<complex> conj(const Matrix<complex>& A)
  {
    Matrix<complex> result(A.size());

    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      result.data()[k] = std::conj(A.data()[k]);

    return result;
  }

  Matrix<complex>&& conj(Matrix<complex>&& A)
  {
    const size_t n = A.numel();
    for (size_t k = 0; k < n; ++k)
      A.data()[k] = std::conj(A.data()[k]);

    return std::move(A);
  }

  Matrix<complex> fft(const Matrix<complex>& signal)
  {
    return fft(Matrix<complex>(signal));
  }

  Matrix<complex>&& fft(Matrix<complex>&& signal)
  {
    const size_t N = signal.numel();

    if (N & (N - 1))
    {
      PINAKAS_ERROR("the number of elements in 'signal' must be a power of 2");
      return std::move(signal);
    }
    
    size_t k = N; // Current stage size
    size_t n; // Size of butterfly operations
    double thetaT = M_PI / static_cast<double>(N); // Angle for twiddle factor
    complex phiT = complex(std::cos(thetaT), -std::sin(thetaT)); // Twiddle factor for the first stage

    // radix-2 decimation-in-frequency variation of the Cooley-Tukey fft algorithm
    while (k > 1)
    {
      n = k;
      k /= 2; // halve stage size
      phiT *= phiT; // Square the twiddle factor for the next stage
      complex twiddle_factor = 1; // Initialize the twiddle factor for the current stage

      // butterfly operations
      for (size_t l = 0; l < k; ++l)
      {
        for (size_t a = l; a < N; a += n)
        {
          size_t b = a + k;
          complex temporary = signal.data()[a] - signal.data()[b];
          signal.data()[a] += signal.data()[b];
          signal.data()[b]  = temporary * twiddle_factor;
        }
        // Update the twiddle factor for the next butterfly operation
        twiddle_factor *= phiT;
      }
    }

    // re-order frequency bins
    const size_t bits_to_reverse = static_cast<size_t>(std::log2(N));
    for (size_t a = 0; a < N; ++a)
    {
      // b = bit reversal of a
      size_t b = a;
      b = ((b & 0xAAAAAAAA) >> 1) | ((b & 0x55555555) << 1);
      b = ((b & 0xCCCCCCCC) >> 2) | ((b & 0x33333333) << 2);
      b = ((b & 0xF0F0F0F0) >> 4) | ((b & 0x0F0F0F0F) << 4);
      b = ((b & 0xFF00FF00) >> 8) | ((b & 0x00FF00FF) << 8);
      b = ((b >> 16) | (b << 16)) >> (32 - bits_to_reverse);
      
      // swap elements
      if (b > a)
      {
        std::swap(signal.data()[a], signal.data()[b]);
      }
    }

    return std::move(signal);
  }

  // Matrix<complex> ifft(const Matrix<complex>& spectrum)
  // {
  //   return conj(fft(conj(spectrum))) / spectrum.numel();
  // }

  // Matrix<complex> ifft(Matrix<complex>&& spectrum)
  // {
  //   return conj(fft(conj(std::move(spectrum)))) / spectrum.numel();
  // }
}
