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
// -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string& op);
// -------------------------------------------------------------------------------
  Matrix<double>& operator+=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator+(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator+(const Matrix<double>& A, const Random range);
  Matrix<double>&& operator+(Matrix<double>&& A, const Random range);
  Matrix<double> operator+(const Random range, const Matrix<double>& A);
  Matrix<double>&& operator+(const Random range, Matrix<double>&& A);
  Matrix<double>&& operator+(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator+(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator+(Matrix<double>&& A, Matrix<double>&& B);
  Matrix<double>& operator+=(Matrix<double>& A, const double B);
  Matrix<double> operator+(const Matrix<double>& A, const double B);
  Matrix<double>&& operator+(Matrix<double>&& A, const double B);
  Matrix<double> operator+(const double A, const Matrix<double>& B);
  Matrix<double>&& operator+(const double A, Matrix<double>&& B);
  inline Matrix<double> operator+(const Matrix<double>& A);
  inline Matrix<double>&& operator+(Matrix<double>&& A);
// -------------------------------------------------------------------------------
  Matrix<double>& operator*=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator*(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double>&& operator*(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator*(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator*(Matrix<double>&& A, Matrix<double>&& B);
  Matrix<double>& operator*=(Matrix<double>& A, const double B);
  Matrix<double> operator*(const Matrix<double>& A, const double B);
  Matrix<double>&& operator*(Matrix<double>&& A, const double B);
  Matrix<double> operator*(const double A, const Matrix<double>& B);
  Matrix<double>&& operator*(const double A, Matrix<double>&& B);
// -------------------------------------------------------------------------------
  Matrix<double>& operator-=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator-(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double>&& operator-(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator-(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator-(Matrix<double>&& A, Matrix<double>&& B);
  Matrix<double>& operator-=(Matrix<double>& A, const double B);
  Matrix<double> operator-(const Matrix<double>& A, const double B);
  Matrix<double>&& operator-(Matrix<double>&& A, const double B);
  Matrix<double> operator-(const double A, const Matrix<double>& B);
  Matrix<double>&& operator-(const double A, Matrix<double>&& B);
  Matrix<double> operator-(const Matrix<double>& A);
  Matrix<double>&& operator-(Matrix<double>&& A);
// -------------------------------------------------------------------------------
  Matrix<double>& operator/=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator/(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double>&& operator/(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator/(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator/(Matrix<double>&& A, Matrix<double>&& B);
  Matrix<double>& operator/=(Matrix<double>& A, const double B);
  Matrix<double> operator/(const Matrix<double>& A, const double B);
  Matrix<double>&& operator/(Matrix<double>&& A, const double B);
  Matrix<double> operator/(const double A, const Matrix<double>& B);
  Matrix<double>&& operator/(const double A, Matrix<double>&& B);
// -------------------------------------------------------------------------------
  Matrix<double>& operator^=(Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> operator^(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double>&& operator^(const Matrix<double>& A, Matrix<double>&& B);
  Matrix<double>&& operator^(Matrix<double>&& A, const Matrix<double>& B);
  Matrix<double>&& operator^(Matrix<double>&& A, Matrix<double>&& B);
  Matrix<double>& operator^=(Matrix<double>& A, const double B);
  Matrix<double> operator^(const Matrix<double>& A, const double B);
  Matrix<double>&& operator^(Matrix<double>&& A, const double B);
  Matrix<double> operator^(const double A, const Matrix<double>& B);
  Matrix<double>&& operator^(const double A, Matrix<double>&& B);
// -------------------------------------------------------------------------------
  Matrix<double> floor(const Matrix<double>& A);
  Matrix<double>&& floor(Matrix<double>&& A);
  Matrix<double> round(const Matrix<double>& A);
  Matrix<double>&& round(Matrix<double>&& A);
  Matrix<double> ceil(Matrix<double>& A);
  Matrix<double>&& ceil(Matrix<double>&& A);
// -------------------------------------------------------------------------------
  Matrix<double> mul(const Matrix<double>& A, const Matrix<double>& B);
// -------------------------------------------------------------------------------
  Matrix<double> transpose(const Matrix<double>& A);
  Matrix<double> reshape(const Matrix<double>& A, const size_t M, const size_t N);
// -------------------------------------------------------------------------------
  double min(const Matrix<double>& matrix);
  double min(const Slice<double>& column);
  double max(const Matrix<double>& matrix);
  double min(const Slice<double>& column);
// -------------------------------------------------------------------------------
  double sum(const Matrix<double>& matrix);
  double prod(const Matrix<double>& matrix);
  double avg(const Matrix<double>& A);
  double rms(const Matrix<double> &A);
  double geo(const Matrix<double>& A);
  Matrix<double> orthogonalize(Matrix<double> A);
  std::unique_ptr<Matrix<double>[]> QR(Matrix<double> A);
  Matrix<double> div(const Matrix<double>& A, Matrix<double> B);
  std::unique_ptr<Matrix<double>[]> linearize(const Matrix<double>& data_x, const Matrix<double>& data_y);
  Matrix<double> linspace(const double x1, const double x2, const size_t N);
  Matrix<double> linspace(const double x1, const double x2, const size_t N, Keyword::Column);
  Matrix<double> iota(const size_t N);
  Matrix<double> reverse(const Matrix<double>& A);
  Matrix<double>&& reverse(Matrix<double>&& A);
  Matrix<double> diff(const Matrix<double>& A, size_t n = 1);
  Matrix<double> conv(const Matrix<double>& A, const Matrix<double>& B);
  Matrix<double> corr(const Matrix<double> &A, const Matrix<double> &B);
  Matrix<double> corr(const Matrix<double> &A);
  Matrix<double> Rxx(const Matrix<double> &A);
  Matrix<double> Rxx(const Matrix<double> &A, const size_t K);
  Matrix<double> lpc(const Matrix<double> &A, const size_t p);
  Matrix<double> toeplitz(const Matrix<double>& A, size_t K);
  Matrix<double> blackman(const size_t N);
  Matrix<double> blackman(const Matrix<double>& signal);
  Matrix<double>&& blackman(Matrix<double>&& signal);
  Matrix<double> hamming(const size_t N);
  Matrix<double> hamming(const Matrix<double>& signal);
  Matrix<double>&& hamming(Matrix<double>&& signal);
  Matrix<double> hann(const size_t N);
  Matrix<double> hann(const Matrix<double>& signal);
  Matrix<double>&& hann(Matrix<double>&& signal);
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed);
  double sinc(const double x);
  Matrix<double> sin(Matrix<double>& A);
  Matrix<double>&& sin(Matrix<double>&& A);
  Matrix<double> sinc(Matrix<double>& A);
  Matrix<double>&& sinc(Matrix<double>&& A);
  Matrix<double> sinc_impulse(const size_t length, const double frequency);
  Matrix<double> resample(const Matrix<double>& data, const size_t L, const size_t keep=2, const double alpha=3.5, const bool tail=false);
  void plot(std::string title, List<DataSet> data_sets, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
  void plot(List<std::string> titles, List<DataSet> data_sets, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
  void plot(std::string title, DataSet data_set, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
  double rms(const Matrix<double>& A);
}}
// --Pinakas library: frontend forward declarations-------------------------------
namespace Pinakas { inline namespace Frontend
{
  using Backend::Matrix;
  using Backend::Random;
  namespace Keyword = Backend::Keyword;
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
}}
// --Pinakas library: backend struct and class definitions------------------------
namespace Pinakas { namespace Backend
{
  struct Size {
    size_t M, N, numel;
    inline bool operator==(const Size B) const;
    inline bool operator!=(const Size B) const;
  };

  template<typename T>
  struct Matrix {
    public:
      // destructor
      ~Matrix();
      // empty matrix
      Matrix();
      // copy constructor
      Matrix(const Matrix<T>& matrix);
      // move constructor
      Matrix(Matrix<T>&& matrix);
      // create a matrix MxN
      Matrix(const size_t M, const size_t N);
      //
      Matrix<T>& operator=(const Matrix<T>& other) &;
      Matrix<T>& operator=(Matrix<T>&& other) &;
      Matrix<T>& operator=(const T value) &;
      // indexing
      inline T* operator[](const size_t y) const;
      // bound-checked flat-indexing
      T& operator()(const size_t index) const;
      T& operator()(Keyword::End) const;
      // bound-checked indexing
      T& operator()(const size_t y, const size_t x) const;
      //
      Slice<T> operator()(Keyword::Entire, const size_t n) &;
      Slice<T> operator()(const size_t m, Keyword::Entire) &;
      // return matrix dimensions
      inline Size size(void) const &;
      inline size_t numel(void) const &;
      inline size_t M(void) const &;
      inline size_t N(void) const &;
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
      Iterator<Matrix<T>> begin(void);
      Iterator<Matrix<T>> end(void);
      ConstIterator<Matrix<T>> begin(void) const;
      ConstIterator<Matrix<T>> end(void) const;
  };

  template<typename T>
  class Slice {
    friend struct Matrix<T>;
    public:
      inline T& operator[](size_t index) const &;
      T& operator()(size_t index) const &;
      inline Size size(void) const &;
      inline size_t numel(void) const &;
    private:
      Slice(Matrix<T>& matrix, const size_t n, Keyword::Column);
      Slice(Matrix<T>& matrix, const size_t n, Keyword::Row);
      const Size size_;
      const size_t fixed_;
      const bool col_row_;
      Matrix<T>& matrix_;
  };

  struct Random {
    Random(double min, double max);
    double min_, max_;
  };

  template<typename T>
  class Iterator {
    public:
      Iterator(Matrix<T>& matrix, const size_t index);
      bool operator==(const Iterator& other) const;
      bool operator!=(const Iterator& other) const;
      Iterator& operator++(void);
      T& operator*(void) const;
    private:
      Matrix<T>& matrix;
      size_t index;
  };

  template<typename T>
  class ConstIterator {
    public:
      ConstIterator(const T& matrix, const size_t index);
      bool operator==(const ConstIterator& other) const;
      bool operator!=(const ConstIterator& other) const;
      ConstIterator& operator++();
      const double& operator*() const;
    private:
      const T& matrix;
      size_t index;
  };
}}
// --Pinakas library: operator overloads forward declarations----------------------
namespace Pinakas { namespace Backend
{
  std::ostream& operator<<(std::ostream& ostream, const Matrix<double>& A);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
  std::ostream& operator<<(std::ostream& ostream, const Slice<double>& A);
}}
#endif
