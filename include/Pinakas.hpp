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
  struct Matrix;
  //
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
  typedef std::pair<const Matrix&, const Matrix&> DataSet;
// -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string& op);
// -------------------------------------------------------------------------------
  Matrix& operator+=(Matrix& A, const Matrix& B);
  Matrix operator+(const Matrix& A, const Matrix& B);
  Matrix operator+(const Matrix& A, const Random range);
  Matrix&& operator+(Matrix&& A, const Random range);
  Matrix operator+(const Random range, const Matrix& A);
  Matrix&& operator+(const Random range, Matrix&& A);
  Matrix&& operator+(const Matrix& A, Matrix&& B);
  Matrix&& operator+(Matrix&& A, const Matrix& B);
  Matrix&& operator+(Matrix&& A, Matrix&& B);
  Matrix& operator+=(Matrix& A, const double B);
  Matrix operator+(const Matrix& A, const double B);
  Matrix&& operator+(Matrix&& A, const double B);
  Matrix operator+(const double A, const Matrix& B);
  Matrix&& operator+(const double A, Matrix&& B);
  inline Matrix operator+(const Matrix& A);
  inline Matrix&& operator+(Matrix&& A);
// -------------------------------------------------------------------------------
  Matrix& operator*=(Matrix& A, const Matrix& B);
  Matrix operator*(const Matrix& A, const Matrix& B);
  Matrix&& operator*(const Matrix& A, Matrix&& B);
  Matrix&& operator*(Matrix&& A, const Matrix& B);
  Matrix&& operator*(Matrix&& A, Matrix&& B);
  Matrix& operator*=(Matrix& A, const double B);
  Matrix operator*(const Matrix& A, const double B);
  Matrix&& operator*(Matrix&& A, const double B);
  Matrix operator*(const double A, const Matrix& B);
  Matrix&& operator*(const double A, Matrix&& B);
// -------------------------------------------------------------------------------
  Matrix& operator-=(Matrix& A, const Matrix& B);
  Matrix operator-(const Matrix& A, const Matrix& B);
  Matrix&& operator-(const Matrix& A, Matrix&& B);
  Matrix&& operator-(Matrix&& A, const Matrix& B);
  Matrix&& operator-(Matrix&& A, Matrix&& B);
  Matrix& operator-=(Matrix& A, const double B);
  Matrix operator-(const Matrix& A, const double B);
  Matrix&& operator-(Matrix&& A, const double B);
  Matrix operator-(const double A, const Matrix& B);
  Matrix&& operator-(const double A, Matrix&& B);
  Matrix operator-(const Matrix& A);
  Matrix&& operator-(Matrix&& A);
// -------------------------------------------------------------------------------
  Matrix& operator/=(Matrix& A, const Matrix& B);
  Matrix operator/(const Matrix& A, const Matrix& B);
  Matrix&& operator/(const Matrix& A, Matrix&& B);
  Matrix&& operator/(Matrix&& A, const Matrix& B);
  Matrix&& operator/(Matrix&& A, Matrix&& B);
  Matrix& operator/=(Matrix& A, const double B);
  Matrix operator/(const Matrix& A, const double B);
  Matrix&& operator/(Matrix&& A, const double B);
  Matrix operator/(const double A, const Matrix& B);
  Matrix&& operator/(const double A, Matrix&& B);
// -------------------------------------------------------------------------------
  Matrix& operator^=(Matrix& A, const Matrix& B);
  Matrix operator^(const Matrix& A, const Matrix& B);
  Matrix&& operator^(const Matrix& A, Matrix&& B);
  Matrix&& operator^(Matrix&& A, const Matrix& B);
  Matrix&& operator^(Matrix&& A, Matrix&& B);
  Matrix& operator^=(Matrix& A, const double B);
  Matrix operator^(const Matrix& A, const double B);
  Matrix&& operator^(Matrix&& A, const double B);
  Matrix operator^(const double A, const Matrix& B);
  Matrix&& operator^(const double A, Matrix&& B);
// -------------------------------------------------------------------------------
  Matrix floor(const Matrix& A);
  Matrix&& floor(Matrix&& A);
  Matrix round(const Matrix& A);
  Matrix&& round(Matrix&& A);
  Matrix ceil(Matrix& A);
  Matrix&& ceil(Matrix&& A);
// -------------------------------------------------------------------------------
  Matrix mul(const Matrix& A, const Matrix& B);
// -------------------------------------------------------------------------------
  Matrix transpose(const Matrix& A);
  Matrix reshape(const Matrix& A, const size_t M, const size_t N);
// -------------------------------------------------------------------------------
  double min(const Matrix& matrix);
  double min(const Slice& column);
  double max(const Matrix& matrix);
  double min(const Slice& column);
// -------------------------------------------------------------------------------
  double sum(const Matrix& matrix);
  double prod(const Matrix& matrix);
  double avg(const Matrix& A);
  double rms(const Matrix &A);
  double geo(const Matrix& A);
  Matrix orthogonalize(Matrix A);
  std::unique_ptr<Matrix[]> QR(Matrix A);
  Matrix div(const Matrix& A, Matrix B);
  std::unique_ptr<Matrix[]> linearize(const Matrix& data_x, const Matrix& data_y);
  Matrix linspace(const double x1, const double x2, const size_t N);
  Matrix linspace(const double x1, const double x2, const size_t N, Keyword::Column);
  Matrix iota(const size_t N);
  Matrix reverse(const Matrix& A);
  Matrix&& reverse(Matrix&& A);
  Matrix diff(const Matrix& A, size_t n = 1);
  Matrix conv(const Matrix& A, const Matrix& B);
  Matrix corr(const Matrix &A, const Matrix &B);
  Matrix corr(const Matrix &A);
  Matrix Rxx(const Matrix &A);
  Matrix Rxx(const Matrix &A, const size_t K);
  Matrix lpc(const Matrix &A, const size_t p);
  Matrix toeplitz(const Matrix& A, size_t K);
  Matrix blackman(const size_t N);
  Matrix blackman(const Matrix& signal);
  Matrix&& blackman(Matrix&& signal);
  Matrix hamming(const size_t N);
  Matrix hamming(const Matrix& signal);
  Matrix&& hamming(Matrix&& signal);
  Matrix hann(const size_t N);
  Matrix hann(const Matrix& signal);
  Matrix&& hann(Matrix&& signal);
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed);
  double sinc(const double x);
  Matrix sin(Matrix& A);
  Matrix&& sin(Matrix&& A);
  Matrix sinc(Matrix& A);
  Matrix&& sinc(Matrix&& A);
  Matrix sinc_impulse(const size_t length, const double frequency);
  Matrix resample(const Matrix& data, const size_t L);
  Matrix resample(const Matrix& data, const size_t L, const size_t keep, const double alpha);
  void plot(std::string title, List<DataSet> data_sets, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
  void plot(List<std::string> titles, List<DataSet> data_sets, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
  void plot(std::string title, DataSet data_set, bool persistent = true, bool remove = true, bool pause = false, bool lines = true);
  double rms(const Matrix& A);
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

  struct Matrix {
    public:
      // destructor
      ~Matrix();
      // empty matrix
      Matrix();
      // copy constructor
      Matrix(const Matrix& matrix);
      // move constructor
      Matrix(Matrix&& matrix);
      // create a matrix MxN
      Matrix(const size_t M, const size_t N);
      //
      Matrix& operator=(const Matrix& other) &;
      Matrix& operator=(Matrix&& other) &;
      Matrix& operator=(const double value) &;
      // indexing
      inline double* operator[](const size_t y) const;
      // bound-checked flat-indexing
      double& operator()(const size_t index) const;
      double& operator()(Keyword::End) const;
      // bound-checked indexing
      double& operator()(const size_t y, const size_t x) const;
      //
      Slice operator()(Keyword::Entire, const size_t n) &;
      Slice operator()(const size_t m, Keyword::Entire) &;
      // return matrix dimensions
      inline Size size(void) const &;
      inline size_t numel(void) const &;
      inline size_t M(void) const &;
      inline size_t N(void) const &;
    private:
      // information regarding matrix size
      Size size_;
      // data is a double[M][N] array
      std::unique_ptr<double[]> data_;
      // allocate data_
      friend void allocate(Matrix* matrix, const size_t M, const size_t N);
    public:
      // create a matrix with the same dimensions as 'matrix'
      inline Matrix(const Size size);
      // create a matrix MxN with a specific value
      Matrix(const size_t M, const size_t N, const double value);
      // create a matrix with the same dimensions as 'matrix' with a specific value
      inline Matrix(const Size size, double value);
      // create a matrix MxN random values from a range
      Matrix(const size_t M, const size_t N, Random range);
      // create a matrix with the same dimensions as 'matrix' with random values from a range
      inline Matrix(const Size size, const Random range);
      // create a matrix from specific values
      Matrix(const List<double> values);
      // create a matrix from specific values
      Matrix(const List<const List<const double>> values);
      // join matrix (side-wise)
      Matrix(const List<const Matrix> list);
    public:
      Iterator<Matrix> begin(void);
      Iterator<Matrix> end(void);
      ConstIterator<Matrix> begin(void) const;
      ConstIterator<Matrix> end(void) const;
  };

  class Slice {
    friend struct Matrix;
    public:
      inline double& operator[](size_t index) const &;
      double& operator()(size_t index) const &;
      inline Size size(void) const &;
      inline size_t numel(void) const &;
    private:
      Slice(Matrix& matrix, const size_t n, Keyword::Column);
      Slice(Matrix& matrix, const size_t n, Keyword::Row);
      const Size size_;
      const size_t fixed_;
      const bool col_row_;
      Matrix& matrix_;
  };

  struct Random {
    Random(double min, double max);
    double min_, max_;
  };

  template<typename T>
  class Iterator {
    public:
      Iterator(T& matrix, const size_t index);
      bool operator==(const Iterator& other) const;
      bool operator!=(const Iterator& other) const;
      Iterator& operator++(void);
      double& operator*(void) const;
    private:
      T& matrix;
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
  std::ostream& operator<<(std::ostream& ostream, const Matrix& A);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
  std::ostream& operator<<(std::ostream& ostream, const Slice& A);
}}
#endif
