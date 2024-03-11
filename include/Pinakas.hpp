// --author------------------------------------------------------------------------------
// 
// Justin Asselin (juste-injuste)
// justin.asselin@usherbrooke.ca
// https://github.com/juste-injuste/Pinakas
// 
// --liscence----------------------------------------------------------------------------
// 
// MIT License
// 
// Copyright (c) 2023 Justin Asselin (juste-injuste)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//  
// --versions----------------------------------------------------------------------------
// 
// --notice------------------------------------------------------------------------------
//  Pinākás
//  laz::Matrix;
//
// --inclusion guard---------------------------------------------------------------------
#ifndef PINAKAS_HPP
#define PINAKAS_HPP
// --necessary standard libraries--------------------------------------------------------
#include <iostream>         // for io
#include <iomanip>          // for io formatting
#include <initializer_list> // for std::initializer_list
#include <memory>           // for std::unique_ptr
#include <cmath>            // for math operations
#include <algorithm>        // for std::min, std::max, std::swap
#include <random>           // for random number generators
#include <functional>       // for std::function<>
#include <utility>          // for std::move
#include <stdexcept>        // for exceptions
#include <sstream>          // for string formatting
#include <limits>           // for std::numeric_limits<>
#include <fstream>          // for ofstream
#include <cstdlib>          // for std::system
#include <cstdio>           // for std::remove, std::sprintf
#include <complex>          // for std::complex
#include <type_traits>      // for std::enable_if, std::is_same, std::common_type
#include <thread>
#include <mutex>  // for std::mutex
// --non-essential depencies-------------------------------------------------------------
#include "Chronometro.hpp"
// #include "Parallilos.hpp"
// --Pinakas library: backend forward declaration----------------------------------------
namespace Pinakas
{
  namespace Global
  {
    std::ostream log{std::clog.rdbuf()}; // logging ostream
    std::ostream err{std::cerr.rdbuf()}; // error ostream
    std::ostream wrn{std::clog.rdbuf()}; // warning ostream
  }

  //
  template<typename T>
  using List = std::initializer_list<T>;
  //
  struct Size;
  //
  struct Random;
  //
  template<typename T>
  class Matrix;
  //
  template<typename T>
  class Slice;
  //
  class Range;

  struct Set;

  typedef std::complex<double> complex;

  namespace _backend
  {
    // template<typename T>
    // auto get_random(Random range) -> std::enable_if<std::is_floating_point<T>::value == true, T>;
    
    // template<typename T>
    // auto get_random(Random range) -> std::enable_if<std::is_floating_point<T>::value != true, T>;

    void error_print(const char* caller, const char* message) noexcept
    {
      static std::mutex mtx;
      std::lock_guard<std::mutex> lock{mtx};
      Global::err << "error: " << caller << ": " << message << std::endl;
    }

#   define PINAKAS_ERROR(...)                   \
      [&](const char* caller) noexcept          \
      {                                         \
        static char buffer[255];                \
        sprintf(buffer, __VA_ARGS__);           \
        _backend::error_print(caller, buffer);   \
      }(__func__)

#   define PINAKAS_ERROR_IF(condition, ...) if (condition) PINAKAS_ERROR(__VA_ARGS__)

    void warning_print(const char* caller, const char* message) noexcept
    {
      static std::mutex mtx;
      std::lock_guard<std::mutex> lock{mtx};
      Global::wrn << "warning: " << caller << ": " << message << std::endl;
    }

#   define PINAKAS_WARNING(...)                   \
      [&](const char* caller) noexcept            \
      {                                           \
        static char buffer[255];                  \
        sprintf(buffer, __VA_ARGS__);             \
        _backend::warning_print(caller, buffer);   \
      }(__func__)

#   define PINAKAS_WARNING_IF(condition, ...) if (condition) PINAKAS_WARNING(__VA_ARGS__)        

# if defined(PINAKAS_LOGGING)
    void log_print(const char* caller, const char* message) noexcept
    {
      static std::mutex mtx;
      std::lock_guard<std::mutex> lock{mtx};
      Global::log << caller << ": " << message << std::endl;
    }

#   define PINAKAS_LOG(...)                  \
      [&](const char* caller) noexcept       \
      {                                      \
        static char buffer[255];             \
        sprintf(buffer, __VA_ARGS__);        \
        Backend::log_print(caller, buffer);  \
      }(__func__)

#   define PINAKAS_LOG_IF(condition, ...) if (condition) PINAKAS_LOG(__VA_ARGS__)
# else
#   define PINAKAS_LOG(...)               void(0)
#   define PINAKAS_LOG_IF(condition, ...) void(0)
# endif
  
# if (not defined NDEBUG) or (defined PINAKAS_LOGGING)
#   define PINAKAS_DEBUG_MODE
# endif

    // enables an overload if there is no loss of precision when casting T2 as T1
    template<typename T1, typename T2>
    using if_no_loss = typename std::enable_if<std::is_same<decltype(T1()+T2()), T1>::value, T1>::type;

    template<template<typename> class M, typename T>
    using if_matrixlike = typename std::enable_if<std::is_same<M<T>, Matrix<T>>::value or std::is_same<M<T>, Slice<T>>::value>::type;

    template<typename T1, typename T2>
    auto _add_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto _add_val_inplace(Matrix<T1>& A, T2 B) noexcept       -> Matrix<T1>&;
    template<typename T>
    auto _add_rng_inplace(Matrix<T>& A, Random B) noexcept    -> Matrix<T>&;

    template<typename T1, typename T2>
    auto _add_mat(const Matrix<T1>& A, const Matrix<T2>& B)   -> Matrix<decltype(T1()+T2())>;
    template<typename T1, typename T2>
    auto _add_val(const Matrix<T1>& A, T2 B) noexcept         -> Matrix<decltype(T1()+T2())>;
    template<typename T>
    auto _add_rng(const Matrix<T>& A, Random B) noexcept      -> Matrix<T>;

    template<typename T1, typename T2>
    auto _mul_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto _mul_val_inplace(Matrix<T1>& A, T2 B) noexcept       -> Matrix<T1>&;
    template<typename T>
    auto _mul_rng_inplace(Matrix<T>& A, Random B) noexcept    -> Matrix<T>&;

    template<typename T1, typename T2>
    auto _mul_mat(const Matrix<T1>& A, const Matrix<T2>& B)   -> Matrix<decltype(T1()+T2())>;
    template<typename T1, typename T2>
    auto _mul_val(const Matrix<T1>& A, T2 B) noexcept         -> Matrix<decltype(T1()+T2())>;
    template<typename T>
    auto _mul_rng(const Matrix<T>& A, Random B) noexcept      -> Matrix<T>;

    template<typename T>
    auto _neg_inplace(Matrix<T>& A) noexcept -> Matrix<T>&;
    template<typename T1, typename T2>
    auto _sub_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto _sub_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A) -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto _sub_ll_val_inplace(Matrix<T1>& A, T2 B) noexcept -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto _sub_rl_val_inplace(Matrix<T1>& B, T2 A) noexcept -> Matrix<T1>&;
    template<typename T>
    auto _sub_ll_rng_inplace(Matrix<T>& A, Random B) noexcept -> Matrix<T>&;
    template<typename T>
    auto _sub_rl_rng_inplace(Matrix<T>& B, Random A) noexcept -> Matrix<T>&;

    template<typename T>
    auto _neg_mat(const Matrix<T> A) -> Matrix<T>;
    template<typename T1, typename T2>
    auto _sub_mat(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()-T2())>;
    template<typename T1, typename T2>
    auto _sub_ll_val(const Matrix<T1>& A, T2 B) noexcept -> Matrix<decltype(T1()-T2())>;
    template<typename T1, typename T2>
    auto _sub_rl_val(const Matrix<T1>& B, T2 A) noexcept -> Matrix<decltype(T1()-T2())>;
    template<typename T>
    auto _sub_ll_rng(const Matrix<T>& A, Random B) noexcept -> Matrix<decltype(T()-float())>;
    template<typename T>
    auto _sub_rl_rng(const Matrix<T>& B, Random A) noexcept -> Matrix<decltype(T()-float())>;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& _div_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& _div_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A);
  template<typename T1, typename T2>
  Matrix<T1>& _div_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2>
  Matrix<T1>& _div_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept;
  template<typename T>
  Matrix<T>& _div_ll_rng_inplace(Matrix<T>& A, const Random B) noexcept;
  template<typename T>
  Matrix<T>& _div_rl_rng_inplace(Matrix<T>& B, const Random A) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> _div_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> _div_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> _div_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
  template<typename T1, typename T3 = decltype(T1()+double())>
  Matrix<T3> _div_ll_rng(const Matrix<T1>& A, const Random B) noexcept;
  template<typename T1, typename T3 = decltype(T1()+double())>
  Matrix<T3> _div_rl_rng(const Matrix<T1>& B, const Random A) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& _pow_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& _pow_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A);
  template<typename T1, typename T2>
  Matrix<T1>& _pow_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2>
  Matrix<T1>& _pow_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> _pow_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> _pow_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> _pow_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
// --------------------------------------------------------------------------------------
  template<template<typename> class M, typename T>
  auto floor(const M<T>& A) -> Matrix<T>;
  template<template<typename> class M, typename T>
  auto floor(M<T>&& A) noexcept -> M<T>&&;
  template<template<typename> class M, typename T>
  auto round(const M<T>& A) -> Matrix<T>;
  template<template<typename> class M, typename T>
  auto round(M<T>&& A) noexcept -> M<T>&&;
  template<template<typename> class M, typename T>
  auto ceil(const M<T>& A) -> Matrix<T>;
  template<template<typename> class M, typename T>
  auto ceil(M<T>&& A) noexcept -> M<T>&&;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<double> mul(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T> div(const Matrix<T>& b, Matrix<T> A);
// --------------------------------------------------------------------------------------
  template<typename T = double>
  Matrix<T> linspace(const double x1, const double x2, const size_t N);
  template<typename T = size_t>
  auto iota(size_t N) -> Matrix<T>;
  template<typename T = double>
  Matrix<T> eye(const size_t M, const size_t N);
// --------------------------------------------------------------------------------------
  template<typename T>
  auto transpose(const Matrix<T>& A) -> Matrix<T>;
  template<typename T>
  auto transpose(Matrix<T>&& A) noexcept -> Matrix<T>&&;
  template<typename T>
  auto reshape(const Matrix<T>& A, size_t M, size_t N) -> Matrix<T>;
  template<typename T>
  auto reshape(Matrix<T>&& A, size_t M, size_t N) noexcept -> Matrix<T>&&;
// --------------------------------------------------------------------------------------
  template<template<typename> class M, typename T>
  T min(const M<T>& A) noexcept;
  template<template<typename> class M, typename T>
  T max(const M<T>& A) noexcept;
  template<template<typename> class M, typename T>
  T sum(const M<T>& A) noexcept;
  template<template<typename> class M, typename T>
  double prod(const M<T>& A) noexcept;
  template<template<typename> class M1, typename T>
  double avg(const M1<T>& A) noexcept;
  template<template<typename> class M1, typename T>
  double rms(const M1<T>& A) noexcept;
  template<template<typename> class M1, typename T>
  double geo(const M1<T>& A) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> orthogonalize(Matrix<double> A);
  std::unique_ptr<Matrix<double>[]> qr(Matrix<double> A);
// --------------------------------------------------------------------------------------
  std::unique_ptr<Matrix<double>[]> linearize(const Matrix<double>& data_x, const Matrix<double>& data_y);
  template<template<typename> class M>
  std::unique_ptr<M<double>[]> linearize(M<double>&& data_x, M<double>&& data_y);
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> reverse(const Matrix<T>& A);
  template<typename T>
  Matrix<T> reverse(const Slice<T>& A);
  template<template<typename> class M, typename T>
  M<T>&&    reverse(M<T>&& A) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> diff(const Matrix<double>& A, size_t n = 1);
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  auto conv(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()*T2())>;

  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3>     corr(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T>      corr(const Matrix<T>& A);
  Matrix<double> Rxx(const Matrix<double>& A);
  Matrix<double> Rxx(const Matrix<double>& A, const size_t K);
  Matrix<double> lpc(const Matrix<double>& A, const size_t p);
  Matrix<double> toeplitz(const Matrix<double>& A);
  double newton(std::function<double(double)> function, double tol, size_t max_iteration, double seed) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<double>   cos(const Matrix<T>& A);
  Matrix<double>&& cos(Matrix<double>&& A) noexcept;
  template<typename T>
  Matrix<double>   sin(const Matrix<T>& A);
  Matrix<double>&& sin(Matrix<double>&& A) noexcept;
  template<typename T>
  Matrix<double>   sinc(const Matrix<T>& A);
  Matrix<double>&& sinc(Matrix<double>&& A) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double>   blackman(size_t N);
  Matrix<double>   blackman(const Matrix<double>& signal);
  Matrix<double>&& blackman(Matrix<double>&& signal) noexcept;
  Matrix<double>   hamming(size_t N);
  Matrix<double>   hamming(const Matrix<double>& signal);
  Matrix<double>&& hamming(Matrix<double>&& signal) noexcept;
  Matrix<double>   hann(size_t N);
  Matrix<double>   hann(const Matrix<double>& signal);
  Matrix<double>&& hann(Matrix<double>&& signal) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> sinc_impulse(const size_t length, const double frequency);
  template<typename T>
  Matrix<double> resample(const Matrix<T>& data, size_t L, size_t reflect = 2, float alpha = 3.5, const bool tail = false);
// --------------------------------------------------------------------------------------
  void plot(List<Set> data_sets, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<double> abs(const Matrix<T>& A);
  Matrix<double> real(const Matrix<complex>& A);
  Matrix<double> imag(const Matrix<complex>& A);
  Matrix<complex> conj(const Matrix<complex>& A);
  Matrix<complex>&& conj(Matrix<complex>&& A);
// --------------------------------------------------------------------------------------
  Matrix<complex>   fft(const Matrix<complex>& signal);
  Matrix<complex>&& fft(Matrix<complex>&& signal);
  Matrix<complex>   ifft(const Matrix<complex>& spectrum);
  Matrix<complex>   ifft(Matrix<complex>&& spectrum);
// --Pinakas library: frontend forward declarations--------------------------------------
// --Pinakas library: backend struct and class definitions-------------------------------
  struct Size final
  {
    size_t M, N, numel;
    inline bool operator==(const Size other) const noexcept;
    inline bool operator!=(const Size other) const noexcept;
  };

  struct Indices
  {
    size_t j, i;
  };

  template<typename T>
  class Slice final
  {
  public:
    inline explicit  Slice(T* matrix_data, Size matrix_size, Range rows, Range cols) noexcept;
    inline           Slice(const Slice<T>& other) noexcept;
    inline Slice<T>& operator=(const Slice<T>& other);
    inline Slice<T>& operator=(const Matrix<T>& other);
    inline Slice<T>& operator=(T value);
    
    inline Size     size()  const & noexcept { return size_; }
    inline size_t numel() const & noexcept { return size_.numel; }
    inline size_t M()     const & noexcept { return size_.M; }
    inline size_t N()     const & noexcept { return size_.N; }
    // indexing
    T*       operator[](const size_t j)        noexcept { return matrix_data_ + (j*matrix_M_ + offset_); }
    const T* operator[](const size_t j)  const noexcept { return matrix_data_ + (j*matrix_M_ + offset_); }
    T*       operator[](const Indices idx)       noexcept { return (matrix_data_ + (idx.j*matrix_M_ + offset_))[idx.i]; }
    const T* operator[](const Indices idx) const noexcept { return (matrix_data_ + (idx.j*matrix_M_ + offset_))[idx.i]; }
    // bound-checked flat-indexing
    T&       operator()(signed int k)       noexcept;
    const T& operator()(signed int k) const noexcept;
    // bound-checked indexing
    T&       operator()(signed int j, signed int i)       noexcept;
    const T& operator()(signed int j, signed int i) const noexcept;
  private:
    T* matrix_data_;
    const size_t matrix_M_;
    const size_t offset_;
    const Size size_;     
  private:
    template<typename T0>
    class Iterator
    {
    public:
      explicit Iterator(Slice<T>& slice, const int value) noexcept : slice_(slice), current_(value) {}
      T0&  operator*() const noexcept
      {
        size_t i = current_ % slice_.size_.N;
        size_t j = current_ / slice_.size_.N;
        return slice_.matrix_data_[i + j*slice_.matrix_M_ + slice_.offset_];
      }
      void operator++()                            noexcept { ++current_;}
      bool operator!=(const Iterator& other) const noexcept { return current_ != other.current_; }
    private:
      Slice<T>& slice_;
      signed int current_;
    };
  public:
    Iterator<T>       begin()       noexcept { return Iterator<T>(*this, 0); }
    Iterator<T>       end()         noexcept { return Iterator<T>(*this, size_.numel); }
    Iterator<const T> begin() const noexcept { return Iterator<const T>(*this, 0); }
    Iterator<const T> end()   const noexcept { return Iterator<const T>(*this, size_.numel); }
  friend class Matrix<T>;
  };

  struct Random final
  {
    explicit Random(const double min, const double max) noexcept;
    const double min_,  max_;
  };

  class Range final
  {
  public:
    inline explicit Range(const size_t stop) noexcept;
    inline explicit Range(const int start, const int stop) noexcept;
    inline explicit Range(const int start, const int stop, const size_t step) noexcept;
    int _start;
    int _stop;
    int _step;
  private:
    class Iterator
    {  
    public:
      explicit Iterator(int value_, int step_)           noexcept
        : _current(value_), _step(step_)
      {}
      int      operator*()                       const noexcept { return _current; }
      void     operator++()                            noexcept { _current += _step; }
      bool     operator!=(const Iterator& other) const noexcept
      { return (_step > 0) ? (_current <= other._current) : (_current >= other._current); }
    private:
      int _current;
      const int _step;
    };
  public:
    Iterator begin() const noexcept { return Iterator(_start, _step); }
    Iterator end()   const noexcept { return Iterator(_stop,  _step); }
  };

  template<typename T>
  class Matrix final
  {
  static_assert(std::is_arithmetic<T>::value
    or std::is_same<T, std::complex<float>>::value
    or std::is_same<T, std::complex<double>>::value
    or std::is_same<T, std::complex<long double>>::value,
    "T must be an arithmetic or complex type");
  public:
    // destructor
    ~Matrix() noexcept;
    // empty constructor
    Matrix() noexcept;
    // empty MxN constructor 
    explicit Matrix(const size_t M, const size_t N);
    // random MxN constructor 
    explicit Matrix(const size_t M, const size_t N, Random range);

    // fill
    explicit Matrix(const size_t M, const size_t N, const T value);
    Matrix<T>& operator=(const T value) noexcept;

    // copy
    Matrix(const Matrix<T>& other);
    Matrix(const Slice<T>& other);
    Matrix<T>& operator=(const Matrix<T>& other);
    Matrix<T>& operator=(const Slice<T>& other);

    // move
    Matrix(Matrix<T>&& other) noexcept;
    Matrix<T>& operator=(Matrix<T>&& other) noexcept;
  public:
    // indexing
    inline auto operator[](size_t j)        noexcept ->       T*;
    inline auto operator[](size_t j)  const noexcept -> const T*;
    inline auto operator[](Indices idx)       noexcept ->       T&;
    inline auto operator[](Indices idx) const noexcept -> const T&;
    // bound-checked flat-indexing
    inline       T& operator()(signed k)       noexcept;
    inline const T& operator()(signed k) const noexcept;
    // bound-checked indexing
    inline       T& operator()(signed j, signed i)       noexcept;
    inline const T& operator()(signed j, signed i) const noexcept;
  public:
    Size             size()  const noexcept { return _size; }
    size_t         numel() const noexcept { return _size.numel; }
    size_t         M()     const noexcept { return _size.M; }
    size_t         N()     const noexcept { return _size.N; }
    operator size_t ()     const noexcept { return _size.numel; } 
  private:
    Size _size; // matrix size information
    Size _true_size;
    T*   _data; // T[M * N] array
    
    void _allocate(size_t M, size_t N); // allocates memory to _data
    void _drop(size_t amount) noexcept;
  public:
    // size-based constructor
    inline explicit Matrix(const Size size);
    // size-based constructor with specific value
    inline explicit Matrix(const Size size, T value);
    // size-based constructor with random values from a range
    inline explicit Matrix(const Size size, const Random range);
    // list-based
    Matrix(const List<T> values);
    // 2D list-based
    Matrix(const List<const List<const T>> values);
    // join matrix sideways
    Matrix(const List<const Matrix<T>> list);
  public:
    Slice<T>       operator()(Range rows, Range cols)           noexcept;
    Slice<T>       operator()(Range rows, signed int col)       noexcept { return operator()(rows, Range{col, col}); }
    Slice<T>       operator()(signed int row, Range cols)       noexcept { return operator()(Range{row, row}, cols); }
    Slice<const T> operator()(Range rows, Range cols)     const noexcept;
    Slice<const T> operator()(Range rows, signed int col) const noexcept { return operator()(rows, Range{col, col}); }
    Slice<const T> operator()(signed int row, Range cols) const noexcept { return operator()(Range{row, row}, cols); }
  public: // container named requirements
    using value_type     = T;
    using reference      = T&;
    using iterator       = T*;
    using const_iterator = const T*;
    iterator       begin()       noexcept { return _data; }
    iterator       end()         noexcept { return _data + _size.numel; }
    const_iterator begin() const noexcept { return _data; }
    const_iterator end()   const noexcept { return _data + _size.numel; }
    T*             data()        noexcept { return _data;}
    const T*       data()  const noexcept { return _data;}
  friend class Slice<T>;
  friend Matrix<T>&& transpose<T>(Matrix<T>&& A) noexcept;
  friend Matrix<T>&& reshape<T>(Matrix<T>&& A, size_t M, size_t N) noexcept;
  };

  struct Set final
  {
    Set(const char* name, const Matrix<double>& x, const Matrix<double>& y) noexcept;
    Set(const char* name, const Matrix<double>& y) noexcept;
  private:
    Matrix<double> _x_if_temp;
    Matrix<double> _y_if_temp;
  public:
    const char* const name;
    const Matrix<double>& x;
    const Matrix<double>& y;
  };
// --Pinakas library: operator overloads forward declarations----------------------------
  template<template<typename> class M, typename T, typename = _backend::if_matrixlike<M, T>>
  std::ostream& operator<<(std::ostream& ostream, const M<T>& A);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
//----------------------------------------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> operator+(const Matrix<T>& A) noexcept { return A; }

  template<typename T>
  Matrix<T>&& operator+(Matrix<T>&& A) noexcept { return std::move(A); }

  template<typename T1, typename T2>
  Matrix<T1>& operator+=(Matrix<T1>& A, const Matrix<T2>& B)
  { return _backend::_add_mat_inplace(A, B); }
  
  template<typename T1, typename T2>
  auto operator+(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()+T2())>
  { return _backend::_add_mat(A, B); }

  template<typename T1, typename T2>
  auto operator+(const Matrix<T1>& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&
  { return std::move(_backend::_add_mat_inplace(B, A)); }
  
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, const Matrix<T2>& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_add_mat_inplace(A, B)); }
  
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&
  {  return std::move(_backend::_add_mat_inplace(B, A)); }

  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_add_mat_inplace(A, B)); }
  
  template<typename T>
  auto operator+(Matrix<T>&& A, Matrix<T>&& B) -> Matrix<T>&&
  { return std::move(_backend::_add_mat_inplace(A, B)); }

  template<typename T>
  Matrix<T>& operator+=(Matrix<T>& A, const Random B) noexcept
  { return _backend::_add_rng_inplace(A, B); }

  template<typename T>
  auto operator+(const Matrix<T>& A, Random B) noexcept -> Matrix<T>
  { return _backend::_add_rng(A, B); }

  template<typename T>
  auto operator+(Matrix<T>&& A, Random B) noexcept -> Matrix<T>&&
  { return std::move(_backend::_add_rng_inplace(A, B)); }

  template<typename T>
  auto operator+(Random A, const Matrix<T>& B) noexcept -> Matrix<T>
  { return _backend::_add_rng(B, A); }

  template<typename T>
  auto operator+(Random A, Matrix<T>&& B) noexcept -> Matrix<T>&&
  { return std::move(_backend::_add_rng_inplace(B, A)); }

  template<typename T1, typename T2>
  Matrix<T1>& operator+=(Matrix<T1>& A, const T2 B) noexcept
  { return _backend::_add_val_inplace(A, B); }

  template<typename T1, typename T2>
  auto operator+(const Matrix<T1>& A, const T2 B) noexcept -> Matrix<decltype(T1()+T2())>
  { return _backend::_add_val(A, B); }

  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, const T2 B) noexcept -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_add_val_inplace(A, B)); }

  template<typename T1, typename T2>
  auto operator+(const T1 A, const Matrix<T2>& B) noexcept -> Matrix<decltype(T1()+T2())>
  { return _backend::_add_val(B, A); }

  template<typename T1, typename T2>
  auto operator+(const T1 A, Matrix<T2>&& B) noexcept -> Matrix<_backend::if_no_loss<T2, T1>>&&
  { return std::move(_backend::_add_val_inplace(B, A)); }
//----------------------------------------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator*=(Matrix<T1>& A, const Matrix<T2>& B)
  { return _backend::_mul_mat_inplace(A, B); }
  
  template<typename T1, typename T2>
  auto operator*(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()*T2())>
  { return _backend::_mul_mat(A, B); }

  template<typename T1, typename T2>
  auto operator*(const Matrix<T1>& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&
  { return std::move(_backend::_mul_mat_inplace(B, A)); }
  
  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, const Matrix<T2>& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_mul_mat_inplace(A, B)); }
  
  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&
  {  return std::move(_backend::_mul_mat_inplace(B, A)); }

  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_mul_mat_inplace(A, B)); }
  
  template<typename T>
  auto operator*(Matrix<T>&& A, Matrix<T>&& B) -> Matrix<T>&&
  { return std::move(_backend::_mul_mat_inplace(A, B)); }

  template<typename T>
  Matrix<T>& operator*=(Matrix<T>& A, const Random B) noexcept
  { return _backend::_mul_rng_inplace(A, B); }

  template<typename T>
  auto operator*(const Matrix<T>& A, Random B) noexcept -> Matrix<T>
  { return _backend::_mul_rng(A, B); }

  template<typename T>
  auto operator*(Matrix<T>&& A, Random B) noexcept -> Matrix<T>&&
  { return std::move(_backend::_mul_rng_inplace(A, B)); }

  template<typename T>
  auto operator*(Random A, const Matrix<T>& B) noexcept -> Matrix<T>
  { return _backend::_mul_rng(B, A); }

  template<typename T>
  auto operator*(Random A, Matrix<T>&& B) noexcept -> Matrix<T>&&
  { return std::move(_backend::_mul_rng_inplace(B, A)); }

  template<typename T1, typename T2>
  Matrix<T1>& operator*=(Matrix<T1>& A, const T2 B) noexcept
  { return _backend::_mul_val_inplace(A, B); }

  template<typename T1, typename T2>
  auto operator*(const Matrix<T1>& A, const T2 B) noexcept -> Matrix<decltype(T1()*T2())>
  { return _backend::_mul_val(A, B); }

  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, const T2 B) noexcept -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_mul_val_inplace(A, B)); }

  template<typename T1, typename T2>
  auto operator*(const T1 A, const Matrix<T2>& B) noexcept -> Matrix<decltype(T1()*T2())>
  { return _backend::_mul_val(B, A); }

  template<typename T1, typename T2>
  auto operator*(const T1 A, Matrix<T2>&& B) noexcept -> Matrix<_backend::if_no_loss<T2, T1>>&&
  { return std::move(_backend::_mul_val_inplace(B, A)); } 
//----------------------------------------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> operator-(const Matrix<T>& A) noexcept { return _backend::_neg_mat(A); }

  template<typename T>
  Matrix<T>&& operator-(Matrix<T>&& A) noexcept { return std::move(_backend::_neg_inplace(A)); }

  template<typename T1, typename T2>
  Matrix<T1>& operator-=(Matrix<T1>& A, const Matrix<T2>& B)
  { return _backend::_sub_ll_mat_inplace(A, B); }
  
  template<typename T1, typename T2>
  auto operator-(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()-T2())>
  { return _backend::_sub_mat(A, B); }

  template<typename T1, typename T2>
  auto operator-(const Matrix<T1>& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&
  { return std::move(_backend::_sub_rl_mat_inplace(B, A)); }
  
  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, const Matrix<T2>& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_sub_ll_mat_inplace(A, B)); }
  
  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&
  {  return std::move(_backend::_sub_rl_mat_inplace(B, A)); }

  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_sub_ll_mat_inplace(A, B)); }
  
  template<typename T>
  auto operator-(Matrix<T>&& A, Matrix<T>&& B) -> Matrix<T>&&
  { return std::move(_backend::_sub_ll_mat_inplace(A, B)); }

  template<typename T>
  Matrix<T>& operator-=(Matrix<T>& A, Random B) noexcept
  { return _backend::_sub_ll_rng_inplace(A, B); }

  template<typename T>
  auto operator-(const Matrix<T>& A, Random B) noexcept -> Matrix<T>
  { return _backend::_sub_ll_rng(A, B); }

  template<typename T>
  auto operator-(Matrix<T>&& A, Random B) noexcept -> Matrix<T>&&
  { return std::move(_backend::_sub_ll_rng_inplace(A, B)); }

  template<typename T>
  auto operator-(Random A, const Matrix<T>& B) noexcept -> Matrix<T>
  { return _backend::_sub_rl_rng(B, A); }

  template<typename T>
  auto operator-(Random A, Matrix<T>&& B) noexcept -> Matrix<T>&&
  { return std::move(_backend::_sub_rl_rng_inplace(B, A)); }

  template<typename T1, typename T2>
  Matrix<T1>& operator-=(Matrix<T1>& A, T2 B) noexcept
  { return _backend::_sub_ll_val_inplace(A, B); }

  template<typename T1, typename T2>
  auto operator-(const Matrix<T1>& A, T2 B) noexcept -> Matrix<decltype(T1()-T2())>
  { return _backend::_sub_ll_val(A, B); }

  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, T2 B) noexcept -> Matrix<_backend::if_no_loss<T1, T2>>&&
  { return std::move(_backend::_sub_ll_val_inplace(A, B)); }

  template<typename T1, typename T2>
  auto operator-(T1 A, const Matrix<T2>& B) noexcept -> Matrix<decltype(T1()-T2())>
  { return _backend::_sub_rl_val(B, A); }

  template<typename T1, typename T2>
  auto operator-(T1 A, Matrix<T2>&& B) noexcept -> Matrix<_backend::if_no_loss<T2, T1>>&&
  { return std::move(_backend::_sub_rl_val_inplace(B, A)); }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator/=(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T>& operator/=(Matrix<T>& A, const Random B) noexcept;
  template<typename T1, typename T2>
  Matrix<T1>& operator/=(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> operator/(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T> operator/(const Matrix<T>& A, const Matrix<T>& B);
  template<typename T, typename T3 = decltype(T()+double())>
  Matrix<T3> operator/(const Matrix<T>& A, const Random B) noexcept;
  template<typename T, typename T3 = decltype(T()+double())>
  Matrix<T3> operator/(const Random A, const Matrix<T>& B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> operator/(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> operator/(const T1 A, const Matrix<T2>& B) noexcept;
  template<typename T1, typename T2, typename T3 = _backend::if_no_loss<T2, T1>>
  Matrix<T3>&& operator/(const Matrix<T1>& A, Matrix<T2>&& B);
  template<typename T1, typename T2, typename T3 = _backend::if_no_loss<T1, T2>>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&;
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&;
  template<typename T>
  Matrix<T>&& operator/(Matrix<T>&& A, Matrix<T>&& B);
  template<typename T, typename T3 = _backend::if_no_loss<T, double>>
  Matrix<T3>&& operator/(Matrix<T>&& A, const Random B) noexcept;
  template<typename T, typename T3 = _backend::if_no_loss<T, double>>
  Matrix<T3>&& operator/(const Random A, Matrix<T>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = _backend::if_no_loss<T2, T1>>
  Matrix<T3>&& operator/(const T1 A, Matrix<T2>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = _backend::if_no_loss<T1, T2>>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const T2 B) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator^=(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& operator^=(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> operator^(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> operator^(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> operator^(const T1 A, const Matrix<T2>& B) noexcept;
  template<typename T1, typename T2, typename T3 = _backend::if_no_loss<T2, T1>>
  Matrix<T3>&& operator^(const Matrix<T1>& A, Matrix<T2>&& B);
  template<typename T1, typename T2, typename T3 = _backend::if_no_loss<T1, T2>>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T2, T1>>&&;
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<_backend::if_no_loss<T1, T2>>&&;
  template<typename T>
  Matrix<T>&& operator^(Matrix<T>&& A, Matrix<T>&& B);
  template<typename T1, typename T2, typename T3 = _backend::if_no_loss<T2, T1>>
  Matrix<T3>&& operator^(const T1 A, Matrix<T2>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = _backend::if_no_loss<T1, T2>>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const T2 B) noexcept;
}
#endif
