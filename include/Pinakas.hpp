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
#include <cstdio>           // for std::remove
#include <complex>          // for std::complex
#include <type_traits>      // for std::enable_if, std::is_same, std::common_type
// --non-essential depencies-------------------------------------------------------------
#include "Chronometro.hpp"
// --Pinakas library: backend forward declaration----------------------------------------
namespace Pinakas { namespace Backend
{
  //
  template<typename T>
  using List = std::initializer_list<T>;
  //
  struct Size;
  //
  template<typename T>
  class Matrix;
  //
  template<typename T>
  class Slice;
  // keywords
  namespace Keyword
  {
    const struct Column {} column;
    const struct Row {} row;
    const struct Entire {} entire;
  }
  //
  struct Random;
  //
  class Range;

  template<typename T>
  class Iterator;
  //
  template<typename T>
  class Const_Iterator;
  //
  typedef std::pair<const Matrix<double>, const Matrix<double>> DataSet;
  typedef std::complex<double> complex;
}}
// --Pinakas library: template meta programming------------------------------------------
namespace Pinakas { namespace Backend
{
  // gives the common type T1 and T2 can be implicitely converted to
  template<typename T1, typename T2>
  using appropriate_type = typename std::common_type<T1, T2>::type;

  // enables an overload if there is no loss of precision when casting T2 as T1
  template<typename T1, typename T2>
  using if_no_loss = typename std::enable_if<std::is_same<appropriate_type<T1, T2>, T1>::value, T1>::type;

  // enables an overload if T is a floating type
  template<typename T>
  using if_floating = typename std::enable_if<std::is_floating_point<T>::value, T>::type;

  // enables an overload if T can be converted to double
  template<typename T>
  using convert_to_double = typename std::enable_if<std::is_convertible<T, double>::value, T>::type;
}}
// --------------------------------------------------------------------------------------
namespace Pinakas { namespace Backend
{
  template<typename T1, typename T2>
  Matrix<T1>& add_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& add_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T>
  Matrix<T>& add_rng_inplace(Matrix<T>& A, const Random B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> add_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> add_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> add_rng(const Matrix<T1>& A, const Random B) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& mul_mat_inplace(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& mul_val_inplace(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T>
  Matrix<T>& mul_rng_inplace(Matrix<T>& A, const Random B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> mul_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> mul_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T3 = appropriate_type<T1, double>>
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
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> sub_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> sub_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> sub_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> sub_ll_rng(const Matrix<T1>& A, const Random B) noexcept;
  template<typename T1, typename T3 = appropriate_type<T1, double>>
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
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> div_ll_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> div_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> div_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
  template<typename T1, typename T3 = appropriate_type<T1, double>>
  Matrix<T3> div_ll_rng(const Matrix<T1>& A, const Random B) noexcept;
  template<typename T1, typename T3 = appropriate_type<T1, double>>
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
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> pow_mat(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> pow_ll_val(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> pow_rl_val(const Matrix<T1>& B, const T2 A) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3> floor(const Matrix<T>& A);
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3>&& floor(Matrix<T>&& A) noexcept;
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3> round(const Matrix<T>& A);
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3>&& round(Matrix<T>&& A) noexcept;
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3> ceil(const Matrix<T>& A);
  template<typename T, typename T3 = if_floating<T>>
  Matrix<T3>&& ceil(Matrix<T>&& A) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<double> mul(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T, typename T0 = convert_to_double<T>>
  Matrix<double> div(const Matrix<T>& A, Matrix<double> B);
// --------------------------------------------------------------------------------------
  Matrix<double> linspace(const double x1, const double x2, const size_t N);
  Matrix<size_t> iota(const size_t n);
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> transpose(const Matrix<T>& A);
  template<typename T>
  Matrix<T> reshape(const Matrix<T>& A, const size_t M, const size_t N);
// --------------------------------------------------------------------------------------
  template<typename T>
  T min(const Matrix<T>& matrix) noexcept;
  template<typename T>
  T min(const Slice<T>& slice) noexcept;
  template<typename T>
  T max(const Matrix<T>& matrix) noexcept;
  template<typename T>
  T max(const Slice<T>& slice) noexcept;
  template<typename T>
  T sum(const Matrix<T>& A) noexcept;
  template<typename T>
  double prod(const Matrix<T>& A) noexcept;
  template<typename T>
  double avg(const Matrix<T>& A) noexcept;
  template<typename T>
  double rms(const Matrix<T>& A) noexcept;
  template<typename T>
  double geo(const Matrix<T>& A) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> orthogonalize(Matrix<double> A);
  std::unique_ptr<Matrix<double>[]> QR(Matrix<double> A);
// --------------------------------------------------------------------------------------
  std::unique_ptr<Matrix<double>[]> linearize(const Matrix<double>& data_x, const Matrix<double>& data_y);
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<T> reverse(const Matrix<T>& A);
  template<typename T>
  Matrix<T>&& reverse(Matrix<T>&& A) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> diff(const Matrix<double>& A, size_t n = 1);
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> conv(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> corr(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T> corr(const Matrix<T>& A);
  Matrix<double> Rxx(const Matrix<double>& A);
  Matrix<double> Rxx(const Matrix<double>& A, const size_t K);
  Matrix<double> lpc(const Matrix<double>& A, const size_t p);
  Matrix<double> toeplitz(const Matrix<double>& A);
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed) noexcept;
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
  Matrix<double> blackman(const size_t N);
  Matrix<double> blackman(const Matrix<double>& signal);
  Matrix<double>&& blackman(Matrix<double>&& signal) noexcept;
  Matrix<double> hamming(const size_t N);
  Matrix<double> hamming(const Matrix<double>& signal);
  Matrix<double>&& hamming(Matrix<double>&& signal) noexcept;
  Matrix<double> hann(const size_t N);
  Matrix<double> hann(const Matrix<double>& signal);
  Matrix<double>&& hann(Matrix<double>&& signal) noexcept;
// --------------------------------------------------------------------------------------
  Matrix<double> sinc_impulse(const size_t length, const double frequency);
  Matrix<double> resample(const Matrix<double>& data, const size_t L, const size_t keep=2, const double alpha=3.5, const bool tail=false);
// --------------------------------------------------------------------------------------
  void plot(List<std::string> titles, List<DataSet> data_sets, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
// --------------------------------------------------------------------------------------
  template<typename T>
  Matrix<double> abs(const Matrix<T>& A);
  Matrix<double> real(const Matrix<complex>& A);
  Matrix<double> imag(const Matrix<complex>& A);
  Matrix<complex> conj(const Matrix<complex>& A);
  Matrix<complex>&& conj(Matrix<complex>&& A);
// --------------------------------------------------------------------------------------
  Matrix<complex>&& fft(Matrix<complex>&& signal);
  Matrix<complex> ifft(const Matrix<complex>& spectrum);
  Matrix<complex> ifft(Matrix<complex>&& spectrum);
}}
// --Pinakas library: frontend forward declarations--------------------------------------
namespace Pinakas { inline namespace Frontend
{
  using Backend::Matrix;
  using Backend::Random;
  namespace Keyword = Backend::Keyword;
  using Backend::complex;
  using Backend::Range;
// --------------------------------------------------------------------------------------
  using Backend::floor;
  using Backend::round;
  using Backend::ceil;
// --------------------------------------------------------------------------------------
  using Backend::mul;
// --------------------------------------------------------------------------------------
  using Backend::transpose;
  using Backend::reshape;
// --------------------------------------------------------------------------------------
  using Backend::min;
  using Backend::max;
// --------------------------------------------------------------------------------------
  using Backend::sum;
  using Backend::prod;
  using Backend::avg;
  using Backend::rms;
  using Backend::geo;
  using Backend::orthogonalize;
  using Backend::QR;
  using Backend::div;
  using Backend::linearize;
  using Backend::linspace;
  using Backend::iota;
  using Backend::diff;
  using Backend::conv;
  using Backend::corr;
  using Backend::Rxx;
  using Backend::lpc;
  using Backend::toeplitz;
  using Backend::blackman;
  using Backend::hamming;
  using Backend::hann;
  using Backend::newton;
  using Backend::plot;
  using Backend::resample;
  using Backend::sinc_impulse;
  using Backend::reverse;
  using Backend::fft;
  using Backend::abs;
}}
// --Pinakas library: backend struct and class definitions-------------------------------
namespace Pinakas { namespace Backend
{
  struct Size final {
    size_t M, N, numel;
    inline bool operator==(const Size B) const noexcept;
    inline bool operator!=(const Size B) const noexcept;
  };

  template<typename T>
  class Matrix final {
    public:
      // destructor
      ~Matrix() noexcept;
      // default constructor
      Matrix() noexcept;
      // copy constructor
      Matrix(const Matrix<T>& other);
      // move constructor
      Matrix(Matrix<T>&& other) noexcept;
      // converting constructor
      template<typename T2>
      Matrix(const Matrix<T2>& other);
      // create a matrix MxN
      Matrix(const size_t M, const size_t N);
      // copy assignation
      Matrix<T>& operator=(const Matrix<T>& other) &;
      // move assignation
      Matrix<T>& operator=(Matrix<T>&& other) & noexcept;
      // convertion assignation
      template<typename T2>
      Matrix<T>& operator=(const Matrix<T2>& other) &;
      // fill assignation
      Matrix<T>& operator=(const T value) & noexcept;
      // indexing
      inline T* operator[](const size_t j) const noexcept;
      // bound-checked flat-indexing
      T& operator()(signed int k) const;
      // bound-checked indexing
      T& operator()(signed int j, signed int i) const;
      //
      Slice<T> operator()(Keyword::Entire, const size_t n) & noexcept;
      Slice<T> operator()(const size_t m, Keyword::Entire) & noexcept;
      // return matrix dimensions
      inline Size size(void) const & noexcept;
      inline size_t numel(void) const & noexcept;
      inline size_t M(void) const & noexcept;
      inline size_t N(void) const & noexcept;      
    private:
      // information regarding matrix size
      Size size_;
      // data is a T[M][N] array
      std::unique_ptr<T[]> data_;
      // allocate data_
      template<typename T1>
      friend void allocate(Matrix<T1>* matrix, const size_t M, const size_t N);
    public:
      // create a matrix with the same dimensions as 'matrix'
      inline Matrix(const Size size);
      // create a matrix MxN with a specific value
      Matrix(const size_t M, const size_t N, const T value);
      // create a matrix with the same dimensions as 'matrix' with a specific value
      inline Matrix(const Size size, T value);
      // create a matrix MxN random values from a range
      Matrix(const size_t M, const size_t N, Random range);
      // create a matrix with the same dimensions as 'matrix' with random values from a range
      inline Matrix(const Size size, const Random range);
      // create a matrix from specific values
      Matrix(const List<T> values);
      // create a matrix from specific values
      Matrix(const List<const List<const T>> values);
      // join matrix (side-wise)
      Matrix(const List<const Matrix<T>> list);
    public:
      Matrix<T>& transpose(void) &;
      Matrix<T>&& transpose(void) &&;
      operator size_t (void);
    private:
      class Iterator final {
        public:
          Iterator(Matrix<T>& matrix, const size_t index) noexcept;
          inline bool operator==(const Matrix<T>::Iterator& other) const noexcept;
          inline bool operator!=(const Matrix<T>::Iterator& other) const noexcept;
          Iterator& operator++(void) noexcept;
          inline T& operator*(void) const noexcept;
        private:
          Matrix<T>& matrix;
          size_t index;
      };
      class Const_Iterator final {
        public:
          Const_Iterator(const Matrix<T>& matrix, const size_t index) noexcept;
          inline bool operator==(const Matrix<T>::Const_Iterator& other) const noexcept;
          inline bool operator!=(const Matrix<T>::Const_Iterator& other) const noexcept;
          Const_Iterator& operator++() noexcept;
          inline const T& operator*() const noexcept;
        private:
          const T& matrix;
          size_t index;
      };
    public:
      Iterator begin(void) noexcept;
      Iterator end(void) noexcept;
      Const_Iterator begin(void) const noexcept;
      Const_Iterator end(void) const noexcept;
  };

  template<typename T>
  class Slice final {
    friend class Matrix<T>;
    public:
      inline T& operator[](size_t index) const & noexcept;
      T& operator()(size_t index) const &;
      inline Size size(void) const & noexcept;
      inline size_t numel(void) const & noexcept;
    private:
      Slice(Matrix<T>& matrix, const size_t n, Keyword::Column) noexcept;
      Slice(Matrix<T>& matrix, const size_t n, Keyword::Row) noexcept;
      const Size size_;
      const size_t fixed_;
      const bool col_row_;
      Matrix<T>& matrix_;
  };

  struct Random final {
    Random(const double min, const double max) noexcept;
    const double min_;
    const double max_;
  };

  class Range final {
    public:
      inline explicit Range(const size_t stop) noexcept;
      inline explicit Range(const int start, const int stop) noexcept;
      inline explicit Range(const int start, const int stop, const size_t step) noexcept;
      const int start;
      const int stop;
    private:
      const int step_;
      class Iterator {
        private:
          int current_;
          const int step_;
        public:
          inline explicit Iterator(const int value, const int step) noexcept;
          inline int operator*() const noexcept;
          inline void operator++() noexcept;
          inline bool operator!=(const Iterator& other) const noexcept;
      };
    public:
      inline Iterator begin() const noexcept;
      inline Iterator end() const noexcept;
  };
}}
// --Pinakas library: operator overloads forward declarations----------------------------
namespace Pinakas { namespace Backend
{
  template<typename T>
  std::ostream& operator<<(std::ostream& ostream, const Matrix<T>& A);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
  std::ostream& operator<<(std::ostream& ostream, const Slice<double>& A);
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator+=(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T>& operator+=(Matrix<T>& A, const Random B) noexcept;
  template<typename T1, typename T2>
  Matrix<T1>& operator+=(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator+(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator+(const Matrix<T>& A, const Random B) noexcept;
  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator+(const Random A, const Matrix<T>& B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator+(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator+(const T1 A, const Matrix<T2>& B) noexcept;
  template<typename T>
  Matrix<T>& operator+(const Matrix<T>& A) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator+(const Matrix<T1>& A, Matrix<T2>&& B);
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator+(Matrix<T1>&& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&;
  template<typename T1, typename T2>
  auto operator+(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&;
  template<typename T>
  Matrix<T>&& operator+(Matrix<T>&& A, Matrix<T>&& B);
  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator+(Matrix<T>&& A, const Random B) noexcept;
  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator+(const Random A, Matrix<T>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator+(const T1 A, Matrix<T2>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
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
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator-(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator-(const Matrix<T>& A, const Random B) noexcept;
  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator-(const Random A, const Matrix<T>& B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator-(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator-(const T1 A, const Matrix<T2>& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator-(const Matrix<T1>& A, Matrix<T2>&& B);
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator-(Matrix<T1>&& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&;
  template<typename T1, typename T2>
  auto operator-(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&;
  template<typename T>
  Matrix<T>&& operator-(Matrix<T>&& A, Matrix<T>&& B);
  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator-(Matrix<T>&& A, const Random B) noexcept;
  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator-(const Random A, Matrix<T>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator-(const T1 A, Matrix<T2>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
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
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator*(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator*(const Matrix<T>& A, const Random B) noexcept;
  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator*(const Random A, const Matrix<T>& B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator*(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator*(const T1 A, const Matrix<T2>& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator*(const Matrix<T1>& A, Matrix<T2>&& B);
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator*(Matrix<T1>&& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&;
  template<typename T1, typename T2>
  auto operator*(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&;
  template<typename T>
  Matrix<T>&& operator*(Matrix<T>&& A, Matrix<T>&& B);
  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator*(Matrix<T>&& A, const Random B) noexcept;
  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator*(const Random A, Matrix<T>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator*(const T1 A, Matrix<T2>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator*(Matrix<T1>&& A, const T2 B) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator/=(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T>
  Matrix<T>& operator/=(Matrix<T>& A, const Random B) noexcept;
  template<typename T1, typename T2>
  Matrix<T1>& operator/=(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator/(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator/(const Matrix<T>& A, const Random B) noexcept;
  template<typename T, typename T3 = appropriate_type<T, double>>
  Matrix<T3> operator/(const Random A, const Matrix<T>& B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator/(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator/(const T1 A, const Matrix<T2>& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator/(const Matrix<T1>& A, Matrix<T2>&& B);
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&;
  template<typename T1, typename T2>
  auto operator/(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&;
  template<typename T>
  Matrix<T>&& operator/(Matrix<T>&& A, Matrix<T>&& B);
  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator/(Matrix<T>&& A, const Random B) noexcept;
  template<typename T, typename T3 = if_no_loss<T, double>>
  Matrix<T3>&& operator/(const Random A, Matrix<T>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator/(const T1 A, Matrix<T2>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator/(Matrix<T1>&& A, const T2 B) noexcept;
// --------------------------------------------------------------------------------------
  template<typename T1, typename T2>
  Matrix<T1>& operator^=(Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  Matrix<T1>& operator^=(Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator^(const Matrix<T1>& A, const Matrix<T2>& B);
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator^(const Matrix<T1>& A, const T2 B) noexcept;
  template<typename T1, typename T2, typename T3 = appropriate_type<T1, T2>>
  Matrix<T3> operator^(const T1 A, const Matrix<T2>& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator^(const Matrix<T1>& A, Matrix<T2>&& B);
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const Matrix<T2>& B);
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T2, T1>>&&;
  template<typename T1, typename T2>
  auto operator^(Matrix<T1>&& A, Matrix<T2>&& B) -> Matrix<if_no_loss<T1, T2>>&&;
  template<typename T>
  Matrix<T>&& operator^(Matrix<T>&& A, Matrix<T>&& B);
  template<typename T1, typename T2, typename T3 = if_no_loss<T2, T1>>
  Matrix<T3>&& operator^(const T1 A, Matrix<T2>&& B) noexcept;
  template<typename T1, typename T2, typename T3 = if_no_loss<T1, T2>>
  Matrix<T3>&& operator^(Matrix<T1>&& A, const T2 B) noexcept;
}}
#endif
