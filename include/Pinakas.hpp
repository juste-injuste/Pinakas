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
// Pinākás
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

  namespace Backend
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
        Backend::error_print(caller, buffer);   \
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
        Backend::warning_print(caller, buffer);   \
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
    auto add_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto add_val_inplace(Matrix<T1>& A, T2 B) noexcept       -> Matrix<T1>&;
    template<typename T>
    auto add_rng_inplace(Matrix<T>& A, Random B) noexcept    -> Matrix<T>&;

    template<typename T1, typename T2>
    auto add_mat(const Matrix<T1>& A, const Matrix<T2>& B)   -> Matrix<decltype(T1()+T2())>;
    template<typename T1, typename T2>
    auto add_val(const Matrix<T1>& A, T2 B) noexcept         -> Matrix<decltype(T1()+T2())>;
    template<typename T>
    auto add_rng(const Matrix<T>& A, Random B) noexcept      -> Matrix<T>;

    template<typename T1, typename T2>
    auto mul_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto mul_val_inplace(Matrix<T1>& A, T2 B) noexcept       -> Matrix<T1>&;
    template<typename T>
    auto mul_rng_inplace(Matrix<T>& A, Random B) noexcept    -> Matrix<T>&;

    template<typename T1, typename T2>
    auto mul_mat(const Matrix<T1>& A, const Matrix<T2>& B)   -> Matrix<decltype(T1()+T2())>;
    template<typename T1, typename T2>
    auto mul_val(const Matrix<T1>& A, T2 B) noexcept         -> Matrix<decltype(T1()+T2())>;
    template<typename T>
    auto mul_rng(const Matrix<T>& A, Random B) noexcept      -> Matrix<T>;

    template<typename T>
    auto neg_inplace(Matrix<T>& A) noexcept -> Matrix<T>&;
    template<typename T1, typename T2>
    auto sub_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto sub_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A) -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto sub_ll_val_inplace(Matrix<T1>& A, T2 B) noexcept -> Matrix<T1>&;
    template<typename T1, typename T2>
    auto sub_rl_val_inplace(Matrix<T1>& B, T2 A) noexcept -> Matrix<T1>&;
    template<typename T>
    auto sub_ll_rng_inplace(Matrix<T>& A, Random B) noexcept -> Matrix<T>&;
    template<typename T>
    auto sub_rl_rng_inplace(Matrix<T>& B, Random A) noexcept -> Matrix<T>&;

    template<typename T>
    auto neg_mat(const Matrix<T> A) -> Matrix<T>;
    template<typename T1, typename T2>
    auto sub_mat(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()-T2())>;
    template<typename T1, typename T2>
    auto sub_ll_val(const Matrix<T1>& A, T2 B) noexcept -> Matrix<decltype(T1()-T2())>;
    template<typename T1, typename T2>
    auto sub_rl_val(const Matrix<T1>& B, T2 A) noexcept -> Matrix<decltype(T1()-T2())>;
    template<typename T>
    auto sub_ll_rng(const Matrix<T>& A, Random B) noexcept -> Matrix<decltype(T()-float())>;
    template<typename T>
    auto sub_rl_rng(const Matrix<T>& B, Random A) noexcept -> Matrix<decltype(T()-float())>;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& div_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& div_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A);
  template<typename T1, typename T2>
  Matrix<T1>& div_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2>
  Matrix<T1>& div_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept;
  template<typename T>
  Matrix<T>& div_ll_rng_inplace(Matrix<T>& A, const Random B) noexcept;
  template<typename T>
  Matrix<T>& div_rl_rng_inplace(Matrix<T>& B, const Random A) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> div_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> div_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> div_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
  template<typename T1, typename T3 = decltype(T1()+double())>
  Matrix<T3> div_ll_rng(const Matrix<T1>& A, const Random B) noexcept;
  template<typename T1, typename T3 = decltype(T1()+double())>
  Matrix<T3> div_rl_rng(const Matrix<T1>& B, const Random A) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& pow_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& pow_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A);
  template<typename T1, typename T2>
  Matrix<T1>& pow_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2>
  Matrix<T1>& pow_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> pow_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> pow_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> pow_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
// --------------------------------------------------------------------------------------
  template<template<typename> class M, typename T>
  Matrix<T> floor(const M<T>& A);
  template<template<typename> class M, typename T>
  M<T>&& floor(M<T>&& A) noexcept;
  template<template<typename> class M, typename T>
  Matrix<T> round(const M<T>& A);
  template<template<typename> class M, typename T>
  M<T>&& round(M<T>&& A) noexcept;
  template<template<typename> class M, typename T>
  Matrix<T> ceil(const M<T>& A);
  template<template<typename> class M, typename T>
  M<T>&& ceil(M<T>&& A) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<double> mul(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T> div(const Matrix<T>& b, Matrix<T> A);
// --------------------------------------------------------------------------------------
  template<typename T = double>
  Matrix<T> linspace(const double x1, const double x2, const unsigned N);
  template<typename T = unsigned>
  Matrix<T> iota(const unsigned n);
  template<typename T = double>
  Matrix<T> eye(const unsigned M, const unsigned N);
// --------------------------------------------------------------------------------------
  template<template<typename> class M1, typename T>
  Matrix<T> transpose(const M1<T>& A);
  template<typename T>
  Matrix<T>&& transpose(Matrix<T>&& A);
  template<template<typename> class M1, typename T>
  Matrix<T> reshape(const M1<T>& A, const unsigned M, const unsigned N);
  template<typename T1>
  Matrix<T1>&& reshape(Matrix<T1>&& A, const unsigned M, const unsigned N);
// --------------------------------------------------------------------------------------
  template<template<typename> class M1, typename T>
  T min(const M1<T>& A) noexcept;
  template<template<typename> class M1, typename T>
  T max(const M1<T>& A) noexcept;
  template<template<typename> class M1, typename T>
  T sum(const M1<T>& A) noexcept;
  template<template<typename> class M1, typename T>
  double prod(const M1<T>& A) noexcept;
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
  template<template<typename> class M1, typename T>
  M1<T>&& reverse(M1<T>&& A) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> diff(const Matrix<double>& A, unsigned n = 1);
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> conv(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = decltype(T1()+T2())>
  Matrix<T3> corr(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T> corr(const Matrix<T>& A);
  Matrix<double> rxx(const Matrix<double>& A);
  Matrix<double> rxx(const Matrix<double>& A, const unsigned K);
  Matrix<double> lpc(const Matrix<double>& A, const unsigned p);
  Matrix<double> toeplitz(const Matrix<double>& A);
  double newton(const std::function<double(double)> function, const double tol, const unsigned max_iteration, const double seed) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<double> cos(Matrix<T>& A);
  Matrix<double>&& cos(Matrix<double>&& A) noexcept;
  template<typename T>
  Matrix<double> sin(Matrix<T>& A);
  Matrix<double>&& sin(Matrix<double>&& A) noexcept;
  template<typename T>
  Matrix<double> sinc(Matrix<T>& A);
  Matrix<double>&& sinc(Matrix<double>&& A) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> blackman(const unsigned N);
  Matrix<double> blackman(const Matrix<double>& signal);
  Matrix<double>&& blackman(Matrix<double>&& signal) noexcept;
  Matrix<double> hamming(const unsigned N);
  Matrix<double> hamming(const Matrix<double>& signal);
  Matrix<double>&& hamming(Matrix<double>&& signal) noexcept;
  Matrix<double> hann(const unsigned N);
  Matrix<double> hann(const Matrix<double>& signal);
  Matrix<double>&& hann(Matrix<double>&& signal) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> sinc_impulse(const unsigned length, const double frequency);
  template<typename T>
  Matrix<double> resample(const Matrix<T>& data, unsigned L, unsigned reflect=2, float alpha=3.5, const bool tail=false);
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
      unsigned M, N, numel;
      inline bool operator==(const Size other) const noexcept;
      inline bool operator!=(const Size other) const noexcept;
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
      inline unsigned numel() const & noexcept { return size_.numel; }
      inline unsigned M()     const & noexcept { return size_.M; }
      inline unsigned N()     const & noexcept { return size_.N; }
      // indexing
      T*       operator[](const unsigned j)       noexcept { return matrix_data_ + (j*matrix_M_ + offset_); }
      const T* operator[](const unsigned j) const noexcept { return matrix_data_ + (j*matrix_M_ + offset_); }
      // bound-checked flat-indexing
      T&       operator()(signed int k)       noexcept;
      const T& operator()(signed int k) const noexcept;
      // bound-checked indexing
      T&       operator()(signed int j, signed int i)       noexcept;
      const T& operator()(signed int j, signed int i) const noexcept;
    private:
      T* matrix_data_;
      const unsigned matrix_M_;
      const unsigned offset_;
      const Size size_;     
    private:
      template<typename T0>
      class Iterator
      {
      public:
        explicit Iterator(Slice<T>& slice, const int value) noexcept : slice_(slice), current_(value) {}
        T0&  operator*() const noexcept
        {
          unsigned i = current_ % slice_.size_.N;
          unsigned j = current_ / slice_.size_.N;
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
      const double min_;
      const double max_;
    };

    class Range final
    {
    public:
      inline explicit Range(const unsigned stop) noexcept;
      inline explicit Range(const int start, const int stop) noexcept;
      inline explicit Range(const int start, const int stop, const unsigned step) noexcept;
      int start;
      int stop;
      int step;
    private:
      class Iterator
      {  
      public:
        explicit Iterator(int value, int step)           noexcept : current(value), step(step) {}
        int      operator*()                       const noexcept { return current; }
        void     operator++()                            noexcept { current += step; }
        bool     operator!=(const Iterator& other) const noexcept
        { return (step > 0) ? (current <= other.current) : (current >= other.current); }
      private:
        int current;
        const int step;
      };
    public:
      Iterator begin() const noexcept { return Iterator(start, step); }
      Iterator end()   const noexcept { return Iterator(stop,  step); }
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
      explicit Matrix(const unsigned M, const unsigned N);
      // random MxN constructor 
      explicit Matrix(const unsigned M, const unsigned N, Random range);

      // fill
      explicit Matrix(const unsigned M, const unsigned N, const T value);
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
      inline       T* operator[](unsigned int j)       noexcept { return data_ + (j * size_.N); }
      inline const T* operator[](unsigned int j) const noexcept { return data_ + (j * size_.N); }
      // bound-checked flat-indexing
      inline       T& operator()(signed int k)       noexcept;
      inline const T& operator()(signed int k) const noexcept;
      // bound-checked indexing
      inline       T& operator()(signed int j, signed int i)       noexcept;
      inline const T& operator()(signed int j, signed int i) const noexcept;
    public:
      Size             size()  const noexcept { return size_; }
      unsigned         numel() const noexcept { return size_.numel; }
      unsigned         M()     const noexcept { return size_.M; }
      unsigned         N()     const noexcept { return size_.N; }
      operator unsigned ()     const noexcept { return size_.numel; } 
    private:
      Size size_; // matrix size information
      T*   data_; // T[M * N] array
      
      void allocate(const unsigned M, const unsigned N); // allocates memory to data_
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
      Slice<T>       operator()(Range rows, signed int col)       noexcept { return operator()(rows, {col, col}); }
      Slice<T>       operator()(signed int row, Range cols)       noexcept { return operator()({row, row}, cols); }
      Slice<const T> operator()(Range rows, Range cols)     const noexcept;
      Slice<const T> operator()(Range rows, signed int col) const noexcept { return operator()(rows, {col, col}); }
      Slice<const T> operator()(signed int row, Range cols) const noexcept { return operator()({row, row}, cols); }
    public: // container named requirements
      using value_type     = T;
      using reference      = T&;
      using iterator       = T*;
      using const_iterator = const T*;
      iterator       begin()       noexcept { return data_; }
      iterator       end()         noexcept { return data_ + size_.numel; }
      const_iterator begin() const noexcept { return data_; }
      const_iterator end()   const noexcept { return data_ + size_.numel; }
      T*             data()        noexcept { return data_;}
      const T*       data()  const noexcept { return data_;}
    friend class Slice<T>;
    friend Matrix<T>&& transpose<T>(Matrix<T>&& A);
    friend Matrix<T>&& reshape<T>(Matrix<T>&& A, const unsigned M, const unsigned N);
    };

    struct Set final
    {
      Set(const char* name, const Matrix<double>& x, const Matrix<double>& y) noexcept;
      Set(const char* name, const Matrix<double>& y) noexcept;
    private:
      Matrix<double> x_if_temp;
      Matrix<double> y_if_temp;
    public:
      const char* const name;
      const Matrix<double>& x;
      const Matrix<double>& y;
    };
// --Pinakas library: operator overloads forward declarations----------------------------
    template<template<typename> class M, typename T, typename = Backend::if_matrixlike<M, T>>
    std::ostream& operator<<(std::ostream& ostream, const M<T>& A);
    std::ostream& operator<<(std::ostream& ostream, const Size size);
//----------------------------------------------------------------------------------------------------------------------
    template<typename T>
    Matrix<T> operator+(const Matrix<T>& A) noexcept { return A; }

    template<typename T>
    Matrix<T>&& operator+(Matrix<T>&& A) noexcept { return std::move(A); }

    template<typename T1, typename T2>
    Matrix<T1>& operator+=(Matrix<T1>& A, const Matrix<T2>& B)
    { return Backend::add_mat_inplace(A, B); }
    
    template<typename T1, typename T2>
    auto operator+(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()+T2())>
    { return Backend::add_mat(A, B); }

    template<typename T1, typename T2>
    auto operator+(const Matrix<T1>& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
    { return std::move(Backend::add_mat_inplace(B, A)); }
    
    template<typename T1, typename T2>
    auto operator+(Matrix<T1>&& A, const Matrix<T2>& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::add_mat_inplace(A, B)); }
    
    template<typename T1, typename T2>
    auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
    {  return std::move(Backend::add_mat_inplace(B, A)); }

    template<typename T1, typename T2>
    auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::add_mat_inplace(A, B)); }
    
    template<typename T>
    auto operator+(Matrix<T>&& A, Matrix<T>&& B) -> Matrix<T>&&
    { return std::move(Backend::add_mat_inplace(A, B)); }

    template<typename T>
    Matrix<T>& operator+=(Matrix<T>& A, const Random B) noexcept
    { return Backend::add_rng_inplace(A, B); }

    template<typename T>
    auto operator+(const Matrix<T>& A, Random B) noexcept -> Matrix<T>
    { return Backend::add_rng(A, B); }

    template<typename T>
    auto operator+(Matrix<T>&& A, Random B) noexcept -> Matrix<T>&&
    { return std::move(Backend::add_rng_inplace(A, B)); }

    template<typename T>
    auto operator+(Random A, const Matrix<T>& B) noexcept -> Matrix<T>
    { return Backend::add_rng(B, A); }

    template<typename T>
    auto operator+(Random A, Matrix<T>&& B) noexcept -> Matrix<T>&&
    { return std::move(Backend::add_rng_inplace(B, A)); }

    template<typename T1, typename T2>
    Matrix<T1>& operator+=(Matrix<T1>& A, const T2 B) noexcept
    { return Backend::add_val_inplace(A, B); }

    template<typename T1, typename T2>
    auto operator+(const Matrix<T1>& A, const T2 B) noexcept -> Matrix<decltype(T1()+T2())>
    { return Backend::add_val(A, B); }

    template<typename T1, typename T2>
    auto operator+(Matrix<T1>&& A, const T2 B) noexcept -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::add_val_inplace(A, B)); }

    template<typename T1, typename T2>
    auto operator+(const T1 A, const Matrix<T2>& B) noexcept -> Matrix<decltype(T1()+T2())>
    { return Backend::add_val(B, A); }

    template<typename T1, typename T2>
    auto operator+(const T1 A, Matrix<T2>&& B) noexcept -> Matrix<Backend::if_no_loss<T2, T1>>&&
    { return std::move(Backend::add_val_inplace(B, A)); }
//----------------------------------------------------------------------------------------------------------------------
    template<typename T1, typename T2>
    Matrix<T1>& operator*=(Matrix<T1>& A, const Matrix<T2>& B)
    { return Backend::mul_mat_inplace(A, B); }
    
    template<typename T1, typename T2>
    auto operator*(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()*T2())>
    { return Backend::mul_mat(A, B); }

    template<typename T1, typename T2>
    auto operator*(const Matrix<T1>& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
    { return std::move(Backend::mul_mat_inplace(B, A)); }
    
    template<typename T1, typename T2>
    auto operator*(Matrix<T1>&& A, const Matrix<T2>& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::mul_mat_inplace(A, B)); }
    
    template<typename T1, typename T2>
    auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
    {  return std::move(Backend::mul_mat_inplace(B, A)); }

    template<typename T1, typename T2>
    auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::mul_mat_inplace(A, B)); }
    
    template<typename T>
    auto operator*(Matrix<T>&& A, Matrix<T>&& B) -> Matrix<T>&&
    { return std::move(Backend::mul_mat_inplace(A, B)); }

    template<typename T>
    Matrix<T>& operator*=(Matrix<T>& A, const Random B) noexcept
    { return Backend::mul_rng_inplace(A, B); }

    template<typename T>
    auto operator*(const Matrix<T>& A, Random B) noexcept -> Matrix<T>
    { return Backend::mul_rng(A, B); }

    template<typename T>
    auto operator*(Matrix<T>&& A, Random B) noexcept -> Matrix<T>&&
    { return std::move(Backend::mul_rng_inplace(A, B)); }

    template<typename T>
    auto operator*(Random A, const Matrix<T>& B) noexcept -> Matrix<T>
    { return Backend::mul_rng(B, A); }

    template<typename T>
    auto operator*(Random A, Matrix<T>&& B) noexcept -> Matrix<T>&&
    { return std::move(Backend::mul_rng_inplace(B, A)); }

    template<typename T1, typename T2>
    Matrix<T1>& operator*=(Matrix<T1>& A, const T2 B) noexcept
    { return Backend::mul_val_inplace(A, B); }

    template<typename T1, typename T2>
    auto operator*(const Matrix<T1>& A, const T2 B) noexcept -> Matrix<decltype(T1()*T2())>
    { return Backend::mul_val(A, B); }

    template<typename T1, typename T2>
    auto operator*(Matrix<T1>&& A, const T2 B) noexcept -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::mul_val_inplace(A, B)); }

    template<typename T1, typename T2>
    auto operator*(const T1 A, const Matrix<T2>& B) noexcept -> Matrix<decltype(T1()*T2())>
    { return Backend::mul_val(B, A); }

    template<typename T1, typename T2>
    auto operator*(const T1 A, Matrix<T2>&& B) noexcept -> Matrix<Backend::if_no_loss<T2, T1>>&&
    { return std::move(Backend::mul_val_inplace(B, A)); } 
//----------------------------------------------------------------------------------------------------------------------
    template<typename T>
    Matrix<T> operator-(const Matrix<T>& A) noexcept { return Backend::neg_mat(A); }

    template<typename T>
    Matrix<T>&& operator-(Matrix<T>&& A) noexcept { return std::move(Backend::neg_inplace(A)); }

    template<typename T1, typename T2>
    Matrix<T1>& operator-=(Matrix<T1>& A, const Matrix<T2>& B)
    { return Backend::sub_ll_mat_inplace(A, B); }
    
    template<typename T1, typename T2>
    auto operator-(const Matrix<T1>& A, const Matrix<T2>& B) -> Matrix<decltype(T1()-T2())>
    { return Backend::sub_mat(A, B); }

    template<typename T1, typename T2>
    auto operator-(const Matrix<T1>& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
    { return std::move(Backend::sub_rl_mat_inplace(B, A)); }
    
    template<typename T1, typename T2>
    auto operator-(Matrix<T1>&& A, const Matrix<T2>& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::sub_ll_mat_inplace(A, B)); }
    
    template<typename T1, typename T2>
    auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&
    {  return std::move(Backend::sub_rl_mat_inplace(B, A)); }

    template<typename T1, typename T2>
    auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::sub_ll_mat_inplace(A, B)); }
    
    template<typename T>
    auto operator-(Matrix<T>&& A, Matrix<T>&& B) -> Matrix<T>&&
    { return std::move(Backend::sub_ll_mat_inplace(A, B)); }

    template<typename T>
    Matrix<T>& operator-=(Matrix<T>& A, Random B) noexcept
    { return Backend::sub_ll_rng_inplace(A, B); }

    template<typename T>
    auto operator-(const Matrix<T>& A, Random B) noexcept -> Matrix<T>
    { return Backend::sub_ll_rng(A, B); }

    template<typename T>
    auto operator-(Matrix<T>&& A, Random B) noexcept -> Matrix<T>&&
    { return std::move(Backend::sub_ll_rng_inplace(A, B)); }

    template<typename T>
    auto operator-(Random A, const Matrix<T>& B) noexcept -> Matrix<T>
    { return Backend::sub_rl_rng(B, A); }

    template<typename T>
    auto operator-(Random A, Matrix<T>&& B) noexcept -> Matrix<T>&&
    { return std::move(Backend::sub_rl_rng_inplace(B, A)); }

    template<typename T1, typename T2>
    Matrix<T1>& operator-=(Matrix<T1>& A, T2 B) noexcept
    { return Backend::sub_ll_val_inplace(A, B); }

    template<typename T1, typename T2>
    auto operator-(const Matrix<T1>& A, T2 B) noexcept -> Matrix<decltype(T1()-T2())>
    { return Backend::sub_ll_val(A, B); }

    template<typename T1, typename T2>
    auto operator-(Matrix<T1>&& A, T2 B) noexcept -> Matrix<Backend::if_no_loss<T1, T2>>&&
    { return std::move(Backend::sub_ll_val_inplace(A, B)); }

    template<typename T1, typename T2>
    auto operator-(T1 A, const Matrix<T2>& B) noexcept -> Matrix<decltype(T1()-T2())>
    { return Backend::sub_rl_val(B, A); }

    template<typename T1, typename T2>
    auto operator-(T1 A, Matrix<T2>&& B) noexcept -> Matrix<Backend::if_no_loss<T2, T1>>&&
    { return std::move(Backend::sub_rl_val_inplace(B, A)); }
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
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator/(const Matrix<T1>& A, Matrix<T2>&& B);
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator/(Matrix<T1>&& A, const Matrix<T2>& B);
    template<typename T1, typename T2>
    auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&;
    template<typename T1, typename T2>
    auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&;
    template<typename T>
    Matrix<T>&& operator/(Matrix<T>&& A, Matrix<T>&& B);
    template<typename T, typename T3 = Backend::if_no_loss<T, double>>
    Matrix<T3>&& operator/(Matrix<T>&& A, const Random B) noexcept;
    template<typename T, typename T3 = Backend::if_no_loss<T, double>>
    Matrix<T3>&& operator/(const Random A, Matrix<T>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator/(const T1 A, Matrix<T2>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
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
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator^(const Matrix<T1>& A, Matrix<T2>&& B);
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator^(Matrix<T1>&& A, const Matrix<T2>& B);
    template<typename T1, typename T2>
    auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&;
    template<typename T1, typename T2>
    auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&;
    template<typename T>
    Matrix<T>&& operator^(Matrix<T>&& A, Matrix<T>&& B);
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator^(const T1 A, Matrix<T2>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator^(Matrix<T1>&& A, const T2 B) noexcept;
}
#endif
