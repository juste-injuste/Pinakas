// --author-----------------------------------------------------------------------
// 
// Justin Asselin (juste-injuste)
// justin.asselin@usherbrooke.ca
// https://github.com/juste-injuste/Pinakas
// 
// --liscence---------------------------------------------------------------------
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
// --versions---------------------------------------------------------------------
// 
// --notice-----------------------------------------------------------------------
// Pinākás
// --inclusion guard--------------------------------------------------------------
#ifndef PINAKAS_HPP
#define PINAKAS_HPP
// --necessary standard libraries-------------------------------------------------
#include <iostream>         // for io
#include <iomanip>          // for io formatting
#include <initializer_list> // for std::initializer_list
#include <memory>           // for std::unique_ptr
#include <cmath>            // for math operations
#include <algorithm>        // for std::min, std::max
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
// --non-essential depencies------------------------------------------------------
#include "Chronometro.hpp"
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas { namespace Backend
{
  //
  template<typename T>
  using List = std::initializer_list<T>;
  //
  struct Size;
  //
  template<typename T>
  struct Matrix;
  //
  template<typename T>
  class Slice;
  // keywords
  namespace Keyword
  {
    const struct End {} end;
    const struct Column {} column;
    const struct Row {} row;
    const struct Entire {} entire;
  }
  //
  struct Random;
  //
  template<typename T>
  class Iterator;
  //
  template<typename T>
  class ConstIterator;
  //
  typedef std::pair<const Matrix<double>&, const Matrix<double>&> DataSet;
  typedef std::complex<double> complex;
// -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string& op);
// -------------------------------------------------------------------------------
  Matrix<double>& operator+=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator+(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator+(const Matrix<double>& A, const Random range);
  Matrix<double>&& operator+(Matrix<double>&& A, const Random range) noexcept;
  Matrix<double> operator+(const Random range, const Matrix<double>& A);
  Matrix<double>&& operator+(const Random range, Matrix<double>&& A) noexcept;
  Matrix<double>&& operator+(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator+(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator+(Matrix<double>&& A, Matrix<double>&& B);
  Matrix<double>& operator+=(Matrix<double>& A, const double B) noexcept;
  Matrix<double> operator+(const Matrix<double>& A, const double B);
  Matrix<double>&& operator+(Matrix<double>&& A, const double B) noexcept;
  Matrix<double> operator+(const double A, const Matrix<double>& B);
  Matrix<double>&& operator+(const double A, Matrix<double>&& B) noexcept;
  inline Matrix<double> operator+(const Matrix<double>& A) noexcept;
  inline Matrix<double>&& operator+(Matrix<double>&& A) noexcept;
// -------------------------------------------------------------------------------
  Matrix<double>& operator*=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator*(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double>&& operator*(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator*(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator*(Matrix<double>&& A, Matrix<double>&& B);
  template<typename T>
  Matrix<T>& operator*=(Matrix<T>& A, const double B) noexcept;
  Matrix<double> operator*(const Matrix<double>& A, const double B);
  Matrix<double>&& operator*(Matrix<double>&& A, const double B) noexcept;
  Matrix<double> operator*(const double A, const Matrix<double>& B);
  Matrix<double>&& operator*(const double A, Matrix<double>&& B) noexcept;
// -------------------------------------------------------------------------------
  Matrix<double>& operator-=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator-(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double>&& operator-(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator-(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator-(Matrix<double>&& A, Matrix<double>&& B);
  Matrix<double>& operator-=(Matrix<double>& A, const double B) noexcept;
  Matrix<double> operator-(const Matrix<double>& A, const double B);
  Matrix<double>&& operator-(Matrix<double>&& A, const double B) noexcept;
  Matrix<double> operator-(const double A, const Matrix<double>& B);
  Matrix<double>&& operator-(const double A, Matrix<double>&& B) noexcept;
  Matrix<double> operator-(const Matrix<double>& A);
  Matrix<double>&& operator-(Matrix<double>&& A) noexcept;
// -------------------------------------------------------------------------------
  Matrix<double>& operator/=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator/(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double>&& operator/(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator/(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator/(Matrix<double>&& A, Matrix<double>&& B);
  template<typename T>
  Matrix<T>& operator/=(Matrix<T>& A, const double B) noexcept;
  Matrix<double> operator/(const Matrix<double>& A, const double B);
  Matrix<double>&& operator/(Matrix<double>&& A, const double B) noexcept;
  Matrix<double> operator/(const double A, const Matrix<double>& B);
  Matrix<double>&& operator/(const double A, Matrix<double>&& B) noexcept;
// -------------------------------------------------------------------------------
  Matrix<double>& operator^=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator^(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double>&& operator^(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator^(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator^(Matrix<double>&& A, Matrix<double>&& B);
  Matrix<double>& operator^=(Matrix<double>& A, const double B) noexcept;
  Matrix<double> operator^(const Matrix<double>& A, const double B);
  Matrix<double>&& operator^(Matrix<double>&& A, const double B) noexcept;
  Matrix<double> operator^(const double A, const Matrix<double>& B);
  Matrix<double>&& operator^(const double A, Matrix<double>&& B) noexcept;
// -------------------------------------------------------------------------------
  Matrix<double> floor(const Matrix<double>& A);
  Matrix<double>&& floor(Matrix<double>&& A) noexcept;
  Matrix<double> round(const Matrix<double>& A);
  Matrix<double>&& round(Matrix<double>&& A) noexcept;
  Matrix<double> ceil(Matrix<double>& A);
  Matrix<double>&& ceil(Matrix<double>&& A) noexcept;
// -------------------------------------------------------------------------------
  Matrix<double> mul(const Matrix<double>& A, const Matrix<double>& B);
// -------------------------------------------------------------------------------
  Matrix<double> transpose(const Matrix<double>& A);
  Matrix<double> reshape(const Matrix<double>& A, const size_t M, const size_t N);
// -------------------------------------------------------------------------------
  double min(const Matrix<double>& matrix) noexcept;
  double min(const Slice<double>& column) noexcept;
  double max(const Matrix<double>& matrix) noexcept;
  double min(const Slice<double>& column) noexcept;
// -------------------------------------------------------------------------------
  double sum(const Matrix<double>& matrix) noexcept;
  double prod(const Matrix<double>& matrix) noexcept;
  double avg(const Matrix<double>& A) noexcept;
  double rms(const Matrix<double>& A) noexcept;
  double geo(const Matrix<double>& A) noexcept;
  Matrix<double> orthogonalize(Matrix<double> A);
  std::unique_ptr<Matrix<double>[]> QR(Matrix<double> A);
  Matrix<double> div(const Matrix<double>& A, Matrix<double> B);
  std::unique_ptr<Matrix<double>[]> linearize(const Matrix<double>& data_x, const Matrix<double>& data_y);
  Matrix<double> linspace(const double x1, const double x2, const size_t N);
  Matrix<double> iota(const size_t N);
  Matrix<double> reverse(const Matrix<double>& A);
  Matrix<double>&& reverse(Matrix<double>&& A) noexcept;
  Matrix<double> diff(const Matrix<double>& A, size_t n = 1);
  Matrix<double> conv(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> corr(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> corr(const Matrix<double>& A);
  Matrix<double> Rxx(const Matrix<double>& A);
  Matrix<double> Rxx(const Matrix<double>& A, const size_t K);
  Matrix<double> lpc(const Matrix<double>& A, const size_t p);
  Matrix<double> toeplitz(const Matrix<double>& A);
  Matrix<double> blackman(const size_t N);
  Matrix<double> blackman(const Matrix<double>& signal);
  Matrix<double>&& blackman(Matrix<double>&& signal) noexcept;
  Matrix<double> hamming(const size_t N);
  Matrix<double> hamming(const Matrix<double>& signal);
  Matrix<double>&& hamming(Matrix<double>&& signal) noexcept;
  Matrix<double> hann(const size_t N);
  Matrix<double> hann(const Matrix<double>& signal);
  Matrix<double>&& hann(Matrix<double>&& signal) noexcept;
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed) noexcept;
  double sinc(const double x) noexcept;
  Matrix<double> sin(Matrix<double>& A);
  Matrix<double>&& sin(Matrix<double>&& A) noexcept;
  Matrix<double> sinc(Matrix<double>& A);
  Matrix<double>&& sinc(Matrix<double>&& A) noexcept;
  Matrix<double> sinc_impulse(const size_t length, const double frequency);
  Matrix<double> resample(const Matrix<double>& data, const size_t L, const size_t keep=2, const double alpha=3.5, const bool tail=false);
  void plot(std::string title, List<DataSet> data_sets, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
  void plot(List<std::string> titles, List<DataSet> data_sets, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
  double rms(const Matrix<double>& A) noexcept;
  template<typename T>
  Matrix<double> abs(const Matrix<T>& A);
  Matrix<double>&& abs(Matrix<double>&& A) noexcept;
  Matrix<double> real(const Matrix<complex>& A);
  Matrix<double> imag(const Matrix<complex>& A);
  Matrix<complex> conj(const Matrix<complex>& A);
  Matrix<complex>&& conj(Matrix<complex>&& A) noexcept;
  template<typename T>
  Matrix<complex> fft(const Matrix<T>& signal);
  Matrix<complex>&& fft(Matrix<complex>&& signal) noexcept;
  Matrix<complex> ifft(const Matrix<complex>& spectrum);
  Matrix<complex>&& ifft(Matrix<complex>&& spectrum) noexcept;
}}
// --Pinakas library: frontend forward declarations-------------------------------
namespace Pinakas { inline namespace Frontend
{
  using Backend::Matrix;
  using Backend::Random;
  namespace Keyword = Backend::Keyword;
  using Backend::complex;
// -------------------------------------------------------------------------------
  using Backend::floor;
  using Backend::round;
  using Backend::ceil;
// -------------------------------------------------------------------------------
  using Backend::mul;
// -------------------------------------------------------------------------------
  using Backend::transpose;
  using Backend::reshape;
// -------------------------------------------------------------------------------
  using Backend::min;
  using Backend::max;
// -------------------------------------------------------------------------------
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
// --Pinakas library: backend struct and class definitions------------------------
namespace Pinakas { namespace Backend
{
  struct Size {
    size_t M, N, numel;
    inline bool operator==(const Size B) const noexcept;
    inline bool operator!=(const Size B) const noexcept;
  };

  template<typename T>
  struct Matrix {
    public:
      // destructor
      ~Matrix() noexcept;
      // empty matrix
      Matrix() noexcept;
      // copy constructor
      Matrix(const Matrix<T>& matrix);
      // move constructor
      Matrix(Matrix<T>&& matrix) noexcept;
      // create a matrix MxN
      Matrix(const size_t M, const size_t N);
      //
      template<typename T2>
      Matrix<T>& operator=(const Matrix<T2>& other) &;
      template<typename T2>
      Matrix<T>& operator=(Matrix<T2>&& other) & noexcept;
      Matrix<T>& operator=(const T value) & noexcept;
      // indexing
      inline T* operator[](const size_t y) const noexcept;
      // bound-checked flat-indexing
      T& operator()(const size_t index) const;
      T& operator()(Keyword::End) const noexcept;
      // bound-checked indexing
      T& operator()(const size_t y, const size_t x) const;
      //
      Slice<T> operator()(Keyword::Entire, const size_t n) & noexcept;
      Slice<T> operator()(const size_t m, Keyword::Entire) & noexcept;
      // return matrix dimensions
      inline Size size(void) const & noexcept;
      inline size_t numel(void) const & noexcept;
      inline size_t M(void) const & noexcept;
      inline size_t N(void) const & noexcept;

      template<typename T2>
      operator Matrix<T2> () const;
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
      Iterator<Matrix<T>> begin(void) noexcept;
      Iterator<Matrix<T>> end(void) noexcept;
      ConstIterator<Matrix<T>> begin(void) const noexcept;
      ConstIterator<Matrix<T>> end(void) const noexcept;
  };

  template<typename T>
  class Slice {
    friend struct Matrix<T>;
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

  struct Random {
    Random(double min, double max) noexcept;
    double min_, max_;
  };

  template<typename T>
  class Iterator {
    public:
      Iterator(Matrix<T>& matrix, const size_t index) noexcept;
      inline bool operator==(const Iterator& other) const noexcept;
      inline bool operator!=(const Iterator& other) const noexcept;
      Iterator& operator++(void) noexcept;
      inline T& operator*(void) const noexcept;
    private:
      Matrix<T>& matrix;
      size_t index;
  };

  template<typename T>
  class ConstIterator {
    public:
      ConstIterator(const T& matrix, const size_t index) noexcept;
      inline bool operator==(const ConstIterator& other) const noexcept;
      inline bool operator!=(const ConstIterator& other) const noexcept;
      ConstIterator& operator++() noexcept;
      inline const double& operator*() const noexcept;
    private:
      const T& matrix;
      size_t index;
  };
}}
// --Pinakas library: operator overloads forward declarations----------------------
namespace Pinakas { namespace Backend
{
  template<typename T>
  std::ostream& operator<<(std::ostream& ostream, const Matrix<T>& A);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
  std::ostream& operator<<(std::ostream& ostream, const Slice<double>& A);
}}
#endif
