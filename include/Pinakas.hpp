// --author-----------------------------------------------------------------------
// 
// Justin Asselin (juste-injuste)
// justin.asselin@usherbrooke.ca
// https://github.com/juste-injuste/Pinakas
// 
// --liscence---------------------------------------------------------------------
// 
// MIdouble License
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
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUdouble WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUdouble NOdouble LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENdouble SHALL THE
// AUTHORS OR COPYRIGHdouble HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORdouble OR OTHERWISE, ARISING FROM,
// OUdouble OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
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
#include <chrono>           // for benchmarking
#include <random>           // for random number generators
#include <functional>       // for std::function<>
#include <utility>          // for std::move
#include <stdexcept>        // for exceptions
#include <sstream>          // for string formatting
#include <limits>           // for std::numeric_limits<>
#include <fstream>          // for ofstream
#include <cstdlib>          // for std::system
#include <cstdio>           // for std::remove
#define tic	auto _start = std::chrono::high_resolution_clock::now()	
#define toc	auto _stop = std::chrono::high_resolution_clock::now(); std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(_stop - _start).count() << " ms\n"
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  template<typename T>
  using List = std::initializer_list<T>;
  template<typename T>
  using Pair = std::pair<T, T>;
  typedef double Value;
  struct Size;  
  struct Matrix;
  class Column;
  class Row;
// -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string& op);
// -------------------------------------------------------------------------------
  Matrix& operator+=(Matrix& A, const Matrix& B);
  Matrix operator+(const Matrix& A, const Matrix& B);
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
  double max(const Matrix& matrix);
// -------------------------------------------------------------------------------
  double sum(const Matrix& matrix);
  double prod(const Matrix& matrix);
  Matrix MGS(Matrix A);
  std::unique_ptr<Matrix[]> QR(Matrix A);
  Matrix div(const Matrix& A, Matrix B);
  std::unique_ptr<Matrix[]> linearize(const Matrix& xdata, const Matrix& ydata);
  Matrix linspace(const double x1, const double x2, const size_t N);
  Matrix iota(const size_t N);
  Matrix diff(const Matrix& A, size_t n);
  Matrix conv(const Matrix& A, const Matrix& B);
  Matrix blackman(const size_t L);
  Matrix hamming(const size_t L);
  Matrix hann(const size_t L);
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed);
  void plot(std::string title, List<Pair<const Matrix&>> data_sets, bool persistent = true, bool remove = true, bool pause = false);
}
// --Pinakas library: frontend forward declarations-------------------------------
namespace Pinakas { inline namespace Frontend
{
  using Backend::Matrix;
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
  using Backend::MGS;
  using Backend::QR;
  using Backend::div;
  using Backend::linearize;
  using Backend::linspace;
  using Backend::iota;
  using Backend::diff;
  using Backend::conv;
  using Backend::blackman;
  using Backend::hamming;
  using Backend::hann;
  using Backend::newton;
  using Backend::plot;
}}
// --Pinakas library: backend struct and class definitions------------------------
namespace Pinakas::Backend
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
      // create a matrix with the same dimensions as 'matrix'
      inline Matrix(const Size size);
      // create a matrix MxN with a specific value
      Matrix(const size_t M, const size_t N, const double value);
      // create a matrix with the same dimensions as 'matrix' with a specific value
      inline Matrix(const Size size, double value);
      // create a matrix MxN random values
      Matrix(const size_t M, const size_t N, Pair<double> range);
      // create a matrix with the same dimensions as 'matrix' with random values
      inline Matrix(const Size size, const Pair<double> range);
      // create a matrix from specific values
      Matrix(const List<double> values);
      // create a matrix from specific values
      Matrix(const List<const List<const double>> values);
      // join matrix (side-wise)
      Matrix(const List<const Matrix> list);
      //
      Matrix& operator=(const Matrix& B) &;
      void operator=(const Matrix& B) && = delete;
      Matrix& operator=(Matrix&& B) &;
      void operator=(Matrix&& B) && = delete;
      Matrix& operator=(const double B) &;
      void operator=(const double B) && = delete;
      // indexing
      inline double* operator[](const size_t y) const;
      // bound-checked flat-indexing
      inline double& operator()(const size_t index) const;
      // bound-checked indexing
      inline double& operator()(const size_t y, const size_t x) const;
      //
      inline Size size(void) const &;
      Size size(void) && = delete;
      //
      inline Column col(size_t n) &;
      void col(size_t) && = delete;
      //
      inline Row row(size_t m) &;
      void row(size_t) && = delete;
    private:
      // allocate memory block
      friend void allocate(Matrix* matrix, const size_t M, const size_t N);
      // information regarding matrix size
      Size size_;
      // memory block for data
      std::unique_ptr<char[]> memory_block_;
      // data is a double[M][N] array
      double** data_;
    public:
      class Iterator {
        public:
          Iterator(Matrix& matrix, const size_t index);
          bool operator==(const Iterator& other) const;
          bool operator!=(const Iterator& other) const;
          Iterator& operator++(void);
          double& operator*(void) const;
        private:
          Matrix& matrix;
          size_t index;
      };
      class Const_Iterator {
      public:
          Const_Iterator(const Matrix& matrix, const size_t index);
          bool operator==(const Const_Iterator& other) const;
          bool operator!=(const Const_Iterator& other) const;
          Const_Iterator& operator++();
          const double& operator*() const;
      private:
          const Matrix& matrix;
          size_t index;
      };
      Iterator begin(void);
      Iterator end(void);
      Const_Iterator begin(void) const;
      Const_Iterator end(void) const;
  };

  class Column {
    friend struct Matrix;
    public:
      inline double& operator[](size_t index) const;
      double& operator()(size_t index) const;
      inline Size size(void) const;
    private:
      Column(Matrix& matrix, const size_t n);
      const Size size_;
      const size_t n_;
      Matrix& matrix_;
  };

  class Row {
    friend struct Matrix;
    public:
      inline double& operator[](const size_t index) const;
      double& operator()(const size_t index) const;
      inline Size size(void) const;
    private:
      Row(Matrix& matrix, const size_t m);
      const Size size_;
      const size_t m_;
      Matrix& matrix_;
  };
}
// --Pinakas library: operator overloads forward declarations----------------------
namespace Pinakas::Backend
{
  std::ostream& operator<<(std::ostream& ostream, const Matrix& A);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
  std::ostream& operator<<(std::ostream& ostream, const Column& A);
  std::ostream& operator<<(std::ostream& ostream, const Row& A);
}
#endif
