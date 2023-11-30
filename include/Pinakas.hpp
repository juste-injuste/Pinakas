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
  
    // gives the common type T1 and T2 can be implicitely converted to
    template<typename T1, typename T2>
    using appropriate_type = typename std::common_type<T1, T2>::type;

    // enables an overload if there is no loss of precision when casting T2 as T1
    template<typename T1, typename T2>
    using if_no_loss = typename std::enable_if<std::is_same<appropriate_type<T1, T2>, T1>::value, T1>::type;

    // enables an overload if T is a floating type
    template<typename T>
    using if_floating = typename std::enable_if<std::is_floating_point<T>::value, T>::type;
  }
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& add_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& add_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T>
  Matrix<T>& add_rng_inplace(Matrix<T>& A, const Random B) noexcept;
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> add_mat_sequ(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> add_val_sequ(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T3 = Backend::appropriate_type<T1, double>>
  Matrix<T3> add_rng(const Matrix<T1>& A, const Random B) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& mul_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& mul_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T>
  Matrix<T>& mul_rng_inplace(Matrix<T>& A, const Random B) noexcept;
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> mul_mat_sequ(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> mul_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T3 = Backend::appropriate_type<T1, double>>
  Matrix<T3> mul_rng(const Matrix<T1>& A, const Random B) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& sub_ll_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& sub_rl_mat_inplace(Matrix<T1>& B, const Matrix<T2>& A);
  template<typename T1, typename T2>
  Matrix<T1>& sub_ll_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2>
  Matrix<T1>& sub_rl_val_inplace(Matrix<T1>& B, const T2 A) noexcept;
  template<typename T>
  Matrix<T>& sub_ll_rng_inplace(Matrix<T>& A, const Random B) noexcept;
  template<typename T>
  Matrix<T>& sub_rl_rng_inplace(Matrix<T>& B, const Random A) noexcept;
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> sub_mat_sequ(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> sub_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> sub_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
  template<typename T1, typename T3 = Backend::appropriate_type<T1, double>>
  Matrix<T3> sub_ll_rng(const Matrix<T1>& A, const Random B) noexcept;
  template<typename T1, typename T3 = Backend::appropriate_type<T1, double>>
  Matrix<T3> sub_rl_rng(const Matrix<T1>& B, const Random A) noexcept;
  template<typename T>
  Matrix<T>& negate_inplace(Matrix<T>& A) noexcept;
  template<typename T>
  Matrix<T> negate(const Matrix<T> A);
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
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> div_mat_sequ(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> div_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> div_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
  template<typename T1, typename T3 = Backend::appropriate_type<T1, double>>
  Matrix<T3> div_ll_rng(const Matrix<T1>& A, const Random B) noexcept;
  template<typename T1, typename T3 = Backend::appropriate_type<T1, double>>
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
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> pow_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> pow_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> pow_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> floor(const Matrix<T>& A);
  template<typename T>
  Matrix<T>&& floor(Matrix<T>&& A) noexcept;
  template<typename T>
  Matrix<T> round(const Matrix<T>& A);
  template<typename T>
  Matrix<T>&& round(Matrix<T>&& A) noexcept;
  template<typename T>
  Matrix<T> ceil(const Matrix<T>& A);
  template<typename T>
  Matrix<T>&& ceil(Matrix<T>&& A) noexcept;
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
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
  Matrix<T3> conv(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
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
  Matrix<double> resample(const Matrix<T>& data, const unsigned L, const unsigned keep=2, const double alpha=3.5, const bool tail=false);
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
  Matrix<complex> fft(const Matrix<complex>& signal);
  Matrix<complex>&& fft(Matrix<complex>&& signal);
  Matrix<complex> ifft(const Matrix<complex>& spectrum);
  Matrix<complex> ifft(Matrix<complex>&& spectrum);
// --Pinakas library: frontend forward declarations--------------------------------------
// --Pinakas library: backend struct and class definitions-------------------------------
    struct Size final {
      unsigned M, N, numel;
      inline bool operator==(const Size other) const noexcept;
      inline bool operator!=(const Size other) const noexcept;
    };

    template<typename T>
    class Matrix final
    {
    public:
      // destructor
      ~Matrix() noexcept;
      // empty constructor
      Matrix() noexcept;
      // empty MxN constructor 
      explicit Matrix(const unsigned M, const unsigned N);
      // random MxN constructor 
      explicit Matrix(const unsigned M, const unsigned N, Random range);

      // fill constructor
      explicit Matrix(const unsigned M, const unsigned N, const T value);
      // fill assignation
      Matrix<T>& operator=(const T value) & noexcept;

      // copy constructor
      Matrix(const Matrix<T>& other);
      // copy assignation
      Matrix<T>& operator=(const Matrix<T>& other) &;

      // move constructor
      Matrix(Matrix<T>&& other) noexcept;
      // move assignation
      Matrix<T>& operator=(Matrix<T>&& other) & noexcept;

      // converting constructor
      template<typename T2>
      Matrix(const Matrix<T2>& other);
      // converting assignation
      template<typename T2>
      Matrix<T>& operator=(const Matrix<T2>& other) &;
    public:
      // indexing
      inline T* operator[](const unsigned j) noexcept;
      inline const T* operator[](const unsigned j) const noexcept;

      // bound-checked flat-indexing
      T& operator()(signed int k);
      const T& operator()(signed int k) const;

      // bound-checked indexing
      T& operator()(signed int j, signed int i);
      const T& operator()(signed int j, signed int i) const;
    public:
      // return matrix dimensions
      inline Size size() const & noexcept;
      inline unsigned numel() const & noexcept;
      inline unsigned M() const & noexcept;
      inline unsigned N() const & noexcept;
      inline operator unsigned () const noexcept;     
    private:
      // matrix size information
      Size size_;
      // T[M * N] array
        T* data_;
      // allocates memory to data_
      void allocate(const unsigned M, const unsigned N);
    public:
      // size-based constructor
      inline explicit Matrix(const Size size);
      // size-based constructor with specific value
      inline explicit Matrix(const Size size, T value);
      // size-based constructor with random values from a range
      inline explicit Matrix(const Size size, const Random range);
      // list-based constructor
      Matrix(const List<T> values);
      // 2D list-based constructor
      Matrix(const List<const List<const T>> values);
      // join matrix sideways constructor
      Matrix(const List<const Matrix<T>> list);
    public:
      Slice<T> operator()(Range rows, Range cols) noexcept;
      Slice<T> operator()(Range rows, signed int col) noexcept;
      Slice<T> operator()(signed int row, Range cols) noexcept;
      Slice<const T> operator()(Range rows, Range cols) const noexcept;
      Slice<const T> operator()(Range rows, signed int col) const noexcept;
      Slice<const T> operator()(signed int row, Range cols) const noexcept;
    public: // container named requirements
      using value_type = T;
      using reference  = T&;
      using iterator   = T*;
      using const_iterator = const T*;
      iterator begin(void) noexcept;
      iterator end(void) noexcept;
      const_iterator begin(void) const noexcept;
      const_iterator end(void) const noexcept;
      T* data(void) noexcept {return data_;}
      const T* data(void) const noexcept {return data_;}
    friend class Slice<T>;
    friend Matrix<T>&& transpose<T>(Matrix<T>&& A);
    friend Matrix<T>&& reshape<T>(Matrix<T>&& A, const unsigned M, const unsigned N);
    };

    template<typename T>
    class Slice final
    {
    public:
      inline explicit Slice(T* matrix_data, const Size matrix_size, const Range rows, const Range cols) noexcept;
      inline Slice(const Slice<T>& other) noexcept;
      Slice<T>& operator=(const Slice<T>& other);
      Slice<T>& operator=(const Matrix<T>& other);
      
      inline Size size(void) const & noexcept;
      inline unsigned numel(void) const & noexcept;
      inline unsigned M(void) const & noexcept;
      inline unsigned N(void) const & noexcept;
      // indexing
      inline T* operator[](const unsigned j) noexcept;
      inline const T* operator[](const unsigned j) const noexcept;
      // bound-checked flat-indexing
      T& operator()(signed int k) noexcept;
      const T& operator()(signed int k) const noexcept;
      // bound-checked indexing
      T& operator()(signed int j, signed int i) noexcept;
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
      private:
        Slice<T>& slice_;
        signed int current_;
      public:
        inline explicit Iterator(Slice<T>& slice, const int value) noexcept;
        inline T0& operator*() const noexcept;
        inline void operator++() noexcept;
        inline bool operator!=(const Iterator& other) const noexcept;
      };
    public:
      inline Iterator<T> begin() noexcept;
      inline Iterator<T> end() noexcept;
      inline Iterator<const T> begin() const noexcept;
      inline Iterator<const T> end() const noexcept;
    };

    struct Random final
    {
      Random(const double min, const double max) noexcept;
      const double min_;
      const double max_;
    };

    class Range final
    {
    public:
      inline explicit Range(const unsigned stop) noexcept;
      inline Range(const int start, const int stop) noexcept;
      inline explicit Range(const int start, const int stop, const unsigned step) noexcept;
      int start;
      int stop;
      int step;
    private:
      class Iterator
      {
      public:
        inline explicit Iterator(const int value, const int step) noexcept;
        inline int operator*() const noexcept;
        inline void operator++() noexcept;
        inline bool operator!=(const Iterator& other) const noexcept;
      private:
        int current;
        const int step;
      };
    public:
      inline Iterator begin() const noexcept;
      inline Iterator end() const noexcept;
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

    template<template<typename> class M1, typename T>
    std::ostream& operator<<(std::ostream& ostream, const M1<T>& A);
    std::ostream& operator<<(std::ostream& ostream, const Size size);
  // --------------------------------------------------------------------------------------
    template<typename T1, typename T2>
    Matrix<T1>& operator+=(Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T>
    Matrix<T>& operator+=(Matrix<T>& A, const Random B) noexcept;
    template<typename T1, typename T2>
    Matrix<T1>& operator+=(Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator+(const Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T>
    Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B);
    template<typename T, typename T3 = Backend::appropriate_type<T, double>>
    Matrix<T3> operator+(const Matrix<T>& A, const Random B) noexcept;
    template<typename T, typename T3 = Backend::appropriate_type<T, double>>
    Matrix<T3> operator+(const Random A, const Matrix<T>& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator+(const Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator+(const T1 A, const Matrix<T2>& B) noexcept;
    template<typename T>
    Matrix<T>& operator+(const Matrix<T>& A) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator+(const Matrix<T1>& A, Matrix<T2>&& B);
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator+(Matrix<T1>&& A, const Matrix<T2>& B);
    template<typename T1, typename T2>
    auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&;
    template<typename T1, typename T2>
    auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&;
    template<typename T>
    Matrix<T>&& operator+(Matrix<T>&& A, Matrix<T>&& B);
    template<typename T, typename T3 = Backend::if_no_loss<T, double>>
    Matrix<T3>&& operator+(Matrix<T>&& A, const Random B) noexcept;
    template<typename T, typename T3 = Backend::if_no_loss<T, double>>
    Matrix<T3>&& operator+(const Random A, Matrix<T>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator+(const T1 A, Matrix<T2>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator+(Matrix<T1>&& A, const T2 B) noexcept;
    template<typename T>
    Matrix<T>&& operator+(Matrix<T>&& A) noexcept;
  // --------------------------------------------------------------------------------------
    template<typename T1, typename T2>
    Matrix<T1>& operator-=(Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T>
    Matrix<T>& operator-=(Matrix<T>& A, const Random B) noexcept;
    template<typename T1, typename T2>
    Matrix<T1>& operator-=(Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator-(const Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T>
    Matrix<T> operator-(const Matrix<T>& A, const Matrix<T>& B);
    template<typename T, typename T3 = Backend::appropriate_type<T, double>>
    Matrix<T3> operator-(const Matrix<T>& A, const Random B) noexcept;
    template<typename T, typename T3 = Backend::appropriate_type<T, double>>
    Matrix<T3> operator-(const Random A, const Matrix<T>& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator-(const Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator-(const T1 A, const Matrix<T2>& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator-(const Matrix<T1>& A, Matrix<T2>&& B);
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator-(Matrix<T1>&& A, const Matrix<T2>& B);
    template<typename T1, typename T2>
    auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&;
    template<typename T1, typename T2>
    auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&;
    template<typename T>
    Matrix<T>&& operator-(Matrix<T>&& A, Matrix<T>&& B);
    template<typename T, typename T3 = Backend::if_no_loss<T, double>>
    Matrix<T3>&& operator-(Matrix<T>&& A, const Random B) noexcept;
    template<typename T, typename T3 = Backend::if_no_loss<T, double>>
    Matrix<T3>&& operator-(const Random A, Matrix<T>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator-(const T1 A, Matrix<T2>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator-(Matrix<T1>&& A, const T2 B) noexcept;
    template<typename T>
    Matrix<T>& operator-(const Matrix<T>& A);
    template<typename T>
    Matrix<T>&& operator-(Matrix<T>&& A) noexcept;
  // --------------------------------------------------------------------------------------
    template<typename T1, typename T2>
    Matrix<T1>& operator*=(Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T>
    Matrix<T>& operator*=(Matrix<T>& A, const Random B) noexcept;
    template<typename T1, typename T2>
    Matrix<T1>& operator*=(Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator*(const Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T>
    Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B);
    template<typename T, typename T3 = Backend::appropriate_type<T, double>>
    Matrix<T3> operator*(const Matrix<T>& A, const Random B) noexcept;
    template<typename T, typename T3 = Backend::appropriate_type<T, double>>
    Matrix<T3> operator*(const Random A, const Matrix<T>& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator*(const Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator*(const T1 A, const Matrix<T2>& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator*(const Matrix<T1>& A, Matrix<T2>&& B);
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator*(Matrix<T1>&& A, const Matrix<T2>& B);
    template<typename T1, typename T2>
    auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T2, T1>>&&;
    template<typename T1, typename T2>
    auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<Backend::if_no_loss<T1, T2>>&&;
    template<typename T>
    Matrix<T>&& operator*(Matrix<T>&& A, Matrix<T>&& B);
    template<typename T, typename T3 = Backend::if_no_loss<T, double>>
    Matrix<T3>&& operator*(Matrix<T>&& A, const Random B) noexcept;
    template<typename T, typename T3 = Backend::if_no_loss<T, double>>
    Matrix<T3>&& operator*(const Random A, Matrix<T>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T2, T1>>
    Matrix<T3>&& operator*(const T1 A, Matrix<T2>&& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::if_no_loss<T1, T2>>
    Matrix<T3>&& operator*(Matrix<T1>&& A, const T2 B) noexcept;
  // --------------------------------------------------------------------------------------
    template<typename T1, typename T2>
    Matrix<T1>& operator/=(Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T>
    Matrix<T>& operator/=(Matrix<T>& A, const Random B) noexcept;
    template<typename T1, typename T2>
    Matrix<T1>& operator/=(Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator/(const Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T>
    Matrix<T> operator/(const Matrix<T>& A, const Matrix<T>& B);
    template<typename T, typename T3 = Backend::appropriate_type<T, double>>
    Matrix<T3> operator/(const Matrix<T>& A, const Random B) noexcept;
    template<typename T, typename T3 = Backend::appropriate_type<T, double>>
    Matrix<T3> operator/(const Random A, const Matrix<T>& B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator/(const Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
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
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator^(const Matrix<T1>& A, const Matrix<T2>& B);
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
    Matrix<T3> operator^(const Matrix<T1>& A, const T2 B) noexcept;
    template<typename T1, typename T2, typename T3 = Backend::appropriate_type<T1, T2>>
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
