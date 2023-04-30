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
#define tic	auto _start = std::chrono::high_resolution_clock::now()	
#define toc	auto _stop = std::chrono::high_resolution_clock::now(); std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count() << " us\n"
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  template<typename T>
  using List = std::initializer_list<T>;
  //
  typedef double Value;
  //
  struct Size;
  struct Matrix;
// -------------------------------------------------------------------------------
  void validate_size(const Size size_A, const Size size_B, const std::string& op);
// -------------------------------------------------------------------------------
  Matrix& operator+=(Matrix& A, const Matrix& B);
  Matrix operator+(const Matrix& A, const Matrix& B);
  Matrix&& operator+(const Matrix& A, Matrix&& B);
  Matrix&& operator+(Matrix&& A, const Matrix& B);
  Matrix&& operator+(Matrix&& A, Matrix&& B);
  Matrix& operator+=(Matrix& A, const Value B) noexcept;
  Matrix operator+(const Matrix& A, const Value B) noexcept;
  inline Matrix&& operator+(Matrix&& A, const Value B) noexcept;
  inline Matrix operator+(const Value A, const Matrix& B) noexcept;
  inline Matrix&& operator+(const Value A, Matrix&& B) noexcept;
// -------------------------------------------------------------------------------
  Matrix& operator*=(Matrix& A, const Matrix& B);
  Matrix operator*(const Matrix& A, const Matrix& B);
  Matrix&& operator*(const Matrix& A, Matrix&& B);
  Matrix&& operator*(Matrix&& A, const Matrix& B);
  Matrix&& operator*(Matrix&& A, Matrix&& B);
  Matrix& operator*=(Matrix& A, const Value B) noexcept;
  Matrix operator*(const Matrix& A, const Value B) noexcept;
  inline Matrix&& operator*(Matrix&& A, const Value B) noexcept;
  inline Matrix operator*(const Value A, const Matrix& B) noexcept;
  inline Matrix&& operator*(const Value A, Matrix&& B) noexcept;
// -------------------------------------------------------------------------------
  Matrix& operator-=(Matrix& A, const Matrix& B);
  Matrix operator-(const Matrix& A, const Matrix& B);
  Matrix&& operator-(const Matrix& A, Matrix&& B);
  Matrix&& operator-(Matrix&& A, const Matrix& B);
  Matrix&& operator-(Matrix&& A, Matrix&& B);
  inline Matrix& operator-=(Matrix& A, const Value B) noexcept;
  inline Matrix operator-(const Matrix& A, const Value B) noexcept;
  inline Matrix&& operator-(Matrix&& A, const Value B) noexcept;
  Matrix operator-(const Value A, const Matrix& B) noexcept;
  Matrix&& operator-(const Value A, Matrix&& B) noexcept;
// -------------------------------------------------------------------------------
  Matrix& operator/=(Matrix& A, const Matrix& B);
  Matrix operator/(const Matrix& A, const Matrix& B);
  Matrix&& operator/(const Matrix& A, Matrix&& B);
  Matrix&& operator/(Matrix&& A, const Matrix& B);
  Matrix&& operator/(Matrix&& A, Matrix&& B);
  inline Matrix& operator/=(Matrix& A, const Value B) noexcept;
  inline Matrix operator/(const Matrix& A, const Value B) noexcept;
  inline Matrix&& operator/(Matrix&& A, const Value B) noexcept;
  Matrix operator/(const Value A, const Matrix& B) noexcept;
  Matrix&& operator/(const Value A, Matrix&& B) noexcept;
// -------------------------------------------------------------------------------
  Matrix& operator^=(Matrix& A, const Matrix& B);
  Matrix operator^(const Matrix& A, const Matrix& B);
  Matrix&& operator^(const Matrix& A, Matrix&& B);
  Matrix&& operator^(Matrix&& A, const Matrix& B);
  Matrix&& operator^(Matrix&& A, Matrix&& B);
  Matrix& operator^=(Matrix& A, const Value B) noexcept;
  Matrix operator^(const Matrix& A, const Value B) noexcept;
  Matrix&& operator^(Matrix&& A, const Value B) noexcept;
  Matrix operator^(const Value A, const Matrix& B) noexcept;
  Matrix&& operator^(const Value A, Matrix&& B) noexcept;
// -------------------------------------------------------------------------------
  Matrix floor(const Matrix& A);
  Matrix&& floor(Matrix&& A) noexcept;
// -------------------------------------------------------------------------------
  Matrix round(const Matrix& A);
  Matrix&& round(Matrix&& A) noexcept;
// -------------------------------------------------------------------------------
  Matrix ceil(Matrix& A);
  Matrix&& ceil(Matrix&& A) noexcept;
// -------------------------------------------------------------------------------
  Matrix mul(const Matrix& A, const Matrix& B);
// -------------------------------------------------------------------------------
  Matrix transpose(const Matrix& A);
// -------------------------------------------------------------------------------
  Value min(const Matrix& matrix) noexcept;
// -------------------------------------------------------------------------------
  Value max(const Matrix& matrix) noexcept;
// -------------------------------------------------------------------------------
  Value sum(const Matrix& matrix) noexcept;
// -------------------------------------------------------------------------------
  Value prod(const Matrix& matrix) noexcept;
}
// --Pinakas library: frontend forward declarations-------------------------------
namespace Pinakas
{
  using Backend::Matrix;
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
      Matrix(const size_t M, const size_t N, const Value value);
      // create a matrix with the same dimensions as 'matrix' with a specific value
      inline Matrix(const Size size, Value value);
      // create a matrix MxN random values
      Matrix(const size_t M, const size_t N, const Value min, const Value max);
      // create a matrix with the same dimensions as 'matrix' with random values
      inline Matrix(const Size size, const Value min, const Value max);
      // create a matrix from specific values
      Matrix(const List<Value> values);
      // create a matrix from specific values
      Matrix(const List<const List<const Value>> values);
      // join matrix (side-wise)
      Matrix(const List<const Matrix> list);
      //
      Matrix& operator=(const Matrix& B);
      Matrix& operator=(Matrix&& B) noexcept;
      Matrix& operator=(const Value B) noexcept;
      // vector indexing
      inline Value* operator[](const size_t y) const noexcept;
      // bound-checked flat-indexing
      inline Value& operator()(const size_t index) const;
      // bound-checked indexing
      inline Value& operator()(const size_t y, const size_t x) const;
      //
      inline Size size(void) const noexcept;
    private:
      // allocate memory block
      void allocate(const size_t M, const size_t N);
      // information regarding matrix size
      Size size_;
      // memory block for data
      std::unique_ptr<char[]> memory_block_;
      // data is a Value[M][N] array
      Value** data_;
  };
}
// --Pinakas library: operator overloads forward declarations----------------------
namespace Pinakas::Backend
{
  std::ostream& operator<<(std::ostream& ostream, const Matrix& matrix);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
}
#endif
