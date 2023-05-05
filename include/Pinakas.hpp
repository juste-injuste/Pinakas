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
#include <iostream>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <functional>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <limits>
#include <new>
#define tic	auto _start = std::chrono::high_resolution_clock::now()	
#define toc	auto _stop = std::chrono::high_resolution_clock::now(); std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(_stop - _start).count() << " ms\n"
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  template<typename T>
  using List = std::initializer_list<T>;
  //
  typedef double Value;
  //
  struct Size;
  //
  
  struct Matrix;
// -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string& op);
// -------------------------------------------------------------------------------
  
  Matrix& operator+=(Matrix& A, const Matrix& B);
  
  Matrix operator+(const Matrix& A, const Matrix& B);
  
  Matrix&& operator+(const Matrix& A, Matrix&& B);
  
  Matrix&& operator+(Matrix&& A, const Matrix& B);
  
  Matrix&& operator+(Matrix&& A, Matrix&& B);
  
  Matrix& operator+=(Matrix& A, const double B) noexcept;
  
  Matrix operator+(const Matrix& A, const double B) noexcept;
  
  inline Matrix&& operator+(Matrix&& A, const double B) noexcept;
  
  inline Matrix operator+(const double A, const Matrix& B) noexcept;
  
  inline Matrix&& operator+(const double A, Matrix&& B) noexcept;
  
  Matrix operator+(const Matrix& A);
  
  Matrix&& operator+(Matrix&& A) noexcept;
  
// -------------------------------------------------------------------------------
  Matrix& operator*=(Matrix& A, const Matrix& B);
  
  Matrix operator*(const Matrix& A, const Matrix& B);
  
  Matrix&& operator*(const Matrix& A, Matrix&& B);
  
  Matrix&& operator*(Matrix&& A, const Matrix& B);
  
  Matrix&& operator*(Matrix&& A, Matrix&& B);
  
  Matrix& operator*=(Matrix& A, const double B);
  
  Matrix&& operator*=(Matrix&& A, const double B) noexcept;
  
  Matrix& operator*(const Matrix& A, const double B);
  
  inline Matrix&& operator*(Matrix&& A, const double B) noexcept;
  
  inline Matrix operator*(const double A, const Matrix& B) noexcept;
  
  inline Matrix&& operator*(const double A, Matrix&& B) noexcept;
  
// -------------------------------------------------------------------------------
  Matrix& operator-=(Matrix& A, const Matrix& B);
  
  Matrix operator-(const Matrix& A, const Matrix& B);
  
  Matrix&& operator-(const Matrix& A, Matrix&& B);
  
  Matrix&& operator-(Matrix&& A, const Matrix& B);
  
  Matrix&& operator-(Matrix&& A, Matrix&& B);
  
  inline Matrix& operator-=(Matrix& A, const double B) noexcept;
  
  inline Matrix operator-(const Matrix& A, const double B) noexcept;
  
  inline Matrix&& operator-(Matrix&& A, const double B) noexcept;
  
  Matrix operator-(const double A, const Matrix& B) noexcept;
  
  Matrix&& operator-(const double A, Matrix&& B) noexcept;
  
  Matrix operator-(const Matrix& A);
  
  Matrix&& operator-(Matrix&& A) noexcept;
  
// -------------------------------------------------------------------------------
  Matrix& operator/=(Matrix& A, const Matrix& B);
  
  Matrix operator/(const Matrix& A, const Matrix& B);
  
  Matrix&& operator/(const Matrix& A, Matrix&& B);
  
  Matrix&& operator/(Matrix&& A, const Matrix& B);
  
  Matrix&& operator/(Matrix&& A, Matrix&& B);
  
  inline Matrix& operator/=(Matrix& A, const double B) noexcept;
  
  inline Matrix operator/(const Matrix& A, const double B) noexcept;
  
  inline Matrix&& operator/(Matrix&& A, const double B) noexcept;
  
  Matrix operator/(const double A, const Matrix& B) noexcept;
  
  Matrix&& operator/(const double A, Matrix&& B) noexcept;
  
// -------------------------------------------------------------------------------
  Matrix& operator^=(Matrix& A, const Matrix& B);
  
  Matrix operator^(const Matrix& A, const Matrix& B);
  
  Matrix&& operator^(const Matrix& A, Matrix&& B);
  
  Matrix&& operator^(Matrix&& A, const Matrix& B);
  
  Matrix&& operator^(Matrix&& A, Matrix&& B);
  
  Matrix& operator^=(Matrix& A, const double B) noexcept;
  
  Matrix operator^(const Matrix& A, const double B) noexcept;
  
  Matrix&& operator^(Matrix&& A, const double B) noexcept;
  
  Matrix operator^(const double A, const Matrix& B) noexcept;
  
  Matrix&& operator^(const double A, Matrix&& B) noexcept;
// -------------------------------------------------------------------------------
  
  Matrix floor(const Matrix& A);
  
  Matrix&& floor(Matrix&& A) noexcept;
  
  Matrix round(const Matrix& A);
  
  Matrix&& round(Matrix&& A) noexcept;
  
  Matrix ceil(Matrix& A);
  
  Matrix&& ceil(Matrix&& A) noexcept;
// -------------------------------------------------------------------------------
  
  Matrix mul(const Matrix& A, const Matrix& B);
// -------------------------------------------------------------------------------
  
  Matrix transpose(const Matrix& A);
  
  Matrix transpose(Matrix&& A);
// -------------------------------------------------------------------------------
  
  double min(const Matrix& matrix) noexcept;
  
  double max(const Matrix& matrix) noexcept;
// -------------------------------------------------------------------------------
  
  double sum(const Matrix& matrix) noexcept;
  
  double prod(const Matrix& matrix) noexcept;

  std::unique_ptr<Matrix[]> MGS(const Matrix& A);

  Matrix div(const Matrix& A, const Matrix& B);
  Matrix fastdiv(const Matrix& A, Matrix B);
  
  std::unique_ptr<Matrix[]> linearize(const Matrix& xdata, const Matrix& ydata);
  
  std::unique_ptr<Matrix[]> linearize(const Matrix& xdata, Matrix&& ydata);
  
  std::unique_ptr<Matrix[]> linearize(Matrix&& xdata, const Matrix& ydata);
  
  std::unique_ptr<Matrix[]> linearize(Matrix&& xdata, Matrix&& ydata);
  Matrix linspace(const double x1, const double x2, const size_t N);
  Matrix iota(const size_t N);
  
  Matrix diff(const Matrix& A, size_t n);
  
  Matrix conv(const Matrix& A, const Matrix& B);
  Matrix blackman(const size_t L);
  
  Matrix hamming(const size_t L);
  
  Matrix hann(const size_t L);
  
  double newton(const std::function<double(double)> function, const double tol, const size_t max_iteration, const double seed);
}
// --Pinakas library: frontend forward declarations-------------------------------
namespace Pinakas
{
  inline namespace Frontend
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
// -------------------------------------------------------------------------------
    using Backend::min;
    using Backend::max;
// -------------------------------------------------------------------------------
    using Backend::sum;
    using Backend::prod;
    using Backend::MGS;
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
  }
}
// --Pinakas library: backend struct and class definitions------------------------
namespace Pinakas::Backend
{
  struct Size {
    size_t M, N, numel;
    inline bool operator==(const Size B) const noexcept;
    inline bool operator!=(const Size B) const noexcept;
  };

  struct Matrix {
    public:
      // destructor
      ~Matrix() noexcept;
      // empty matrix
      Matrix() noexcept;
      // copy constructor
      Matrix(const Matrix& matrix);
      // move constructor
      Matrix(Matrix&& matrix) noexcept;
      // create a matrix MxN
      Matrix(const size_t M, const size_t N);
      // create a matrix with the same dimensions as 'matrix'
      inline Matrix(const Size size);
      // create a matrix MxN with a specific value
      Matrix(const size_t M, const size_t N, const double value);
      // create a matrix with the same dimensions as 'matrix' with a specific value
      inline Matrix(const Size size, double value);
      // create a matrix MxN random values
      Matrix(const size_t M, const size_t N, const double min, const double max);
      // create a matrix with the same dimensions as 'matrix' with random values
      inline Matrix(const Size size, const double min, const double max);
      // create a matrix from specific values
      Matrix(const List<double> values);
      // create a matrix from specific values
      Matrix(const List<const List<const double>> values);
      // join matrix (side-wise)
      Matrix(const List<const Matrix> list);

      //
      Matrix& operator=(const Matrix& B);
      Matrix& operator=(Matrix&& B) noexcept;
      Matrix& operator=(const double B) noexcept;
      // vector indexing
      inline double* operator[](const size_t y) const noexcept;
      // bound-checked flat-indexing
      inline double& operator()(const size_t index) const;
      // bound-checked indexing
      inline double& operator()(const size_t y, const size_t x) const;
      //
      inline Size size(void) const noexcept;
    private:
      // allocate memory block
      friend void allocate(Matrix* matrix, const size_t M, const size_t N, char* address = nullptr);
      // information regarding matrix size
      Size size_;
      // memory block for data
      std::unique_ptr<char[]> memory_block_;
      // data is a T[M][N] array
      double** data_;
    public:
      class Iterator {
        public:
          Iterator(Matrix& matrix, const size_t index) : matrix(matrix), index(index) {}
          bool operator==(const Iterator& other) const { return index == other.index; }
          bool operator!=(const Iterator& other) const { return !(*this == other); }
          Iterator& operator++() { ++index; return *this; }
          double& operator*() const { return matrix[0][index]; }
        private:
          Matrix& matrix;
          size_t index;
      };
      Iterator begin() { return Iterator(*this, 0); }
      Iterator end()   { return Iterator(*this, size_.numel); }
  };
}
// --Pinakas library: operator overloads forward declarations----------------------
namespace Pinakas::Backend
{
  
  std::ostream& operator<<(std::ostream& ostream, const Matrix& matrix);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
}
#endif
