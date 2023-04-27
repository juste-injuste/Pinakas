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
//
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
#define tic	auto _start = std::chrono::high_resolution_clock::now()	
#define toc	auto _stop = std::chrono::high_resolution_clock::now(); std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count() << " us\n"
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  template<typename T>
  using List = std::initializer_list<T>;
  //
  typedef float Value;
  //
  struct Matrix;
  //
  Matrix operator+(Matrix& A, Matrix& B);
  inline Matrix operator+(Matrix& A, Matrix&& B);
  inline Matrix operator+(Matrix&& A, Matrix& B);
  inline Matrix operator+(Matrix&& A, Matrix&& B);
  //
  Matrix operator+(Matrix& A, Value B);
  inline Matrix operator+(Matrix&& A, Value B);
  inline Matrix operator+(Value A, Matrix& B);
  inline Matrix operator+(Value A, Matrix&& B);
  //
  Matrix operator-(Matrix& A, Matrix& B);
  inline Matrix operator-(Matrix& A, Matrix&& B);
  inline Matrix operator-(Matrix&& A, Matrix& B);
  inline Matrix operator-(Matrix&& A, Matrix&& B);
  //
  Matrix operator-(Matrix& A, Value B);
  inline Matrix operator-(Matrix&& A, Value B);
  inline Matrix operator-(Value A, Matrix& B);
  inline Matrix operator-(Value A, Matrix&& B);
  //
  Matrix operator*(Matrix& A, Matrix& B);
  inline Matrix operator*(Matrix& A, Matrix&& B);
  inline Matrix operator*(Matrix&& A, Matrix& B);
  inline Matrix operator*(Matrix&& A, Matrix&& B);
  //
  Matrix operator*(Matrix& A, Value B);
  inline Matrix operator*(Matrix&& A, Value B);
  inline Matrix operator*(Value A, Matrix& B);
  inline Matrix operator*(Value A, Matrix&& B);
  //
  Matrix operator/(Matrix& A, Matrix& B);
  inline Matrix operator/(Matrix& A, Matrix&& B);
  inline Matrix operator/(Matrix&& A, Matrix& B);
  inline Matrix operator/(Matrix&& A, Matrix&& B);
  //
  Matrix operator/(Matrix& A, Value B);
  inline Matrix operator/(Matrix&& A, Value B);
  inline Matrix operator/(Value A, Matrix& B);
  inline Matrix operator/(Value A, Matrix&& B);
  //
  Matrix floor(Matrix& matrix);
  inline Matrix floor(Matrix&& matrix);
  //
  Matrix round(Matrix& matrix);
  inline Matrix round(Matrix&& matrix);
  //
  Matrix ceil(Matrix& matrix);
  inline Matrix ceil(Matrix&& matrix);
}
// --Pinakas library: frontend forward declarations-------------------------------
namespace Pinakas
{
  using Backend::Matrix;
}
// --Pinakas library: backend struct and class definitions------------------------
namespace Pinakas::Backend
{
  struct Matrix {
    public:
      Matrix();
      //
      Matrix(List<List<Value>> values);
      // create a matrix MxN
      Matrix(size_t M, size_t N);
      // create a matrix MxN with a specific value
      Matrix(size_t M, size_t N, Value value);
      // create a matrix MxN random values
      Matrix(size_t M, size_t N, Value min, Value max);
      // copy a matrix
      Matrix(const Matrix& matrix);
      //

      // find minimum of matrix
      Value min(void) const;
      // find maximum of matrix
      Value max(void) const;
      //
      Matrix& operator=(Matrix& B);
      Matrix& operator=(Matrix&& B);
      Matrix& operator=(Value B);
      // vector indexing
      inline Value* operator[](size_t y) const;
      // bound-checked flat-indexing
      inline Value& operator()(size_t index) const;
      // bound-checked indexing
      inline Value& operator()(size_t y, size_t x) const;
      // information regarding matrix size
      const struct Size {
        size_t M, N, numel;
      } size;
    private:
      // memory block for data
      std::unique_ptr<char[]> memory_block;
      // data is a Value[M][N] array
      Value** data;
  };
}
// --Pinakas library: operator overloads forward declarations----------------------
namespace Pinakas::Backend
{
  Matrix operator+(Matrix& A, Matrix& B);

  std::ostream& operator<<(std::ostream& ostream, const Matrix& matrix);
  std::ostream& operator<<(std::ostream& ostream, const Matrix::Size size);
}
// --Pinakas library: backend struct and class member definitions-----------------
namespace Pinakas::Backend
{
  Matrix::Matrix()
    : // member initialization list
    size{0, 0, 0},
    memory_block(nullptr),
    data(nullptr)
  {}

  Matrix::Matrix(List<List<Value>> values)
    : // member initialization list
    size{values.size(), 0, 0}
  {
    // used throughout constructor
    size_t y, x;

    // dimension validation
    for (List<Value> vector : values) {
      if (size.N && size.N != vector.size()) {
        std::cerr << "vertical dimensions mismatch (" << size.N << " vs " << vector.size() << ")\n";
        const_cast<Size&>(size) = {0, 0, 0};
        return;
      }
      else const_cast<size_t&>(size.N) = vector.size();
    }
    const_cast<size_t&>(size.numel) = size.M*size.N;

    // allocate memory for matrix
    memory_block.reset(new char[(sizeof(Value*) + sizeof(Value) * size.N) * size.M]);

    // get the address of the memory block
    char* address = memory_block.get();

    // validate memory allocation
    if (!address) {
      std::cerr << "! invalid memory allocation !\n";
      exit(0);
    }

    // create rows into memory block
    data = new (address) Value*[size.M];
    // offset address
    address += sizeof(Value*) * size.M;
    
    for (y = 0; y < size.M; ++y) {
      // create columns into memory block
      data[y] = new (address) Value[size.N];
      // offset address
      address += sizeof(Value) * size.N;
    }

    // assign values into matrix
    y = 0;
    for (List<Value> vector : values) {
      x = 0;
      for (Value value : vector) {
        data[y][x] = value;
        ++x;
      }
      ++y;
    }
  }

  Matrix::Matrix(size_t M, size_t N)
    : // member initialization list
    size{M, N, M*N},
    memory_block(new char[(sizeof(Value*) + sizeof(Value) * size.N) * size.M])
  {
    // get the address of the memory block
    char* address = memory_block.get();

    // validate memory allocation
    if (!address) {
      std::cerr << "! invalid memory allocation !\n";
      exit(0);
    }

    // create rows into memory block
    data = new (address) Value*[size.M];
    // offset address
    address += sizeof(Value*) * size.M;
    
    for (size_t y = 0; y < size.M; ++y) {
      // create columns into memory block
      data[y] = new (address) Value[size.N];
      // offset address
      address += sizeof(Value) * size.N;
    }
  }

  Matrix::Matrix(size_t M, size_t N, Value value)
    : // member initialization list
    size{M, N, M*N},
    memory_block(new char[(sizeof(Value*) + sizeof(Value) * size.N) * size.M])
  {
    // get the address of the memory block
    char* address = memory_block.get();

    // validate memory allocation
    if (!address) {
      std::cerr << "! invalid memory allocation !\n";
      exit(0);
    }

    // create rows into memory block
    data = new (address) Value*[size.M];
    // offset address
    address += sizeof(Value*) * size.M;
    
    for (size_t y = 0; y < size.M; ++y) {
      // create columns into memory block
      data[y] = new (address) Value[size.N];
      // offset address
      address += sizeof(Value) * size.N;
    }

    // assign value to matrix
    for (size_t index = 0; index < size.numel; ++index)
      data[0][index] = value;
  }

  Matrix::Matrix(size_t M, size_t N, Value min, Value max)
    : // member initialization list
    size{M, N, M*N},
    memory_block(new char[(sizeof(Value*) + sizeof(Value) * size.N) * size.M])
  {
    // get the address of the memory block
    char* address = memory_block.get();

    // validate memory allocation
    if (!address) {
      std::cerr << "! invalid memory allocation !\n";
      exit(0);
    }

    // create rows into memory block
    data = new (address) Value*[size.M];
    // offset address
    address += sizeof(Value*) * size.M;
    
    for (size_t y = 0; y < size.M; ++y) {
      // create columns into memory block
      data[y] = new (address) Value[size.N];
      // offset address
      address += sizeof(Value) * size.N;
    }

    // random number generator
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> uniform_distribution(min, max);

    // assign random value to matrix
    for (size_t index = 0; index < size.numel; ++index)
      data[0][index] = uniform_distribution(generator);
  }

  // copy a matrix
  Matrix::Matrix(const Matrix& matrix)
    : // member initialization list
    size(matrix.size),
    memory_block(new char[(sizeof(Value*) + sizeof(Value) * size.N) * size.M])
  {
    // get the address of the memory block
    char* address = memory_block.get();

    // validate memory allocation
    if (!address) {
      std::cerr << "! invalid memory allocation !\n";
      exit(0);
    }

    // create rows into memory block
    data = new (address) Value*[size.M];
    // offset address
    address += sizeof(Value*) * size.M;
    
    for (size_t y = 0; y < size.M; ++y) {
      // create columns into memory block
      data[y] = new (address) Value[size.N];
      // offset address
      address += sizeof(Value) * size.N;
    }

    // assign value to matrix
    for (size_t index = 0; index < size.numel; ++index)
      data[0][index] = matrix[0][index];

  }

  Value Matrix::min(void) const
  {
    Value minimum = data[0][0];

    // find minimal value
    for (size_t index = 1; index < size.numel; ++index)
      if (data[0][index] < minimum) minimum = data[0][index];

    return minimum;
  }

  Value Matrix::max(void) const
  {
    Value maximum = data[0][0];

    // find minimal value
    for (size_t index = 1; index < size.numel; ++index)
      if (data[0][index] > maximum) maximum = data[0][index];
      
    return maximum;
  }

  inline Value* Matrix::operator[](size_t index) const
  {
    return data[index];
  }

  inline Value& Matrix::operator()(size_t index) const
  {
    // return value if the index is valid
    if (index < size.N*size.M) return data[0][index];
    // out of bound error message
    std::cerr << '(' << index << ") out of bound " << size.N*size.M << " (dimensions are " << size << ")\n";
    return data[0][0];
  }

  inline Value& Matrix::operator()(size_t y, size_t x) const
  {
    // return value if the index is valid
    if ((y < size.M) && (x < size.N)) return data[y][x];
    // out of bound error messages
    else if (y >= size.M)
      std::cerr << '(' << y << ",_) out of bound " << size.M << " (dimensions are " << size << ")\n";
    else if (x >= size.N)
      std::cerr << "(_," << x << ") out of bound " << size.N << " (dimensions are " << size << ")\n";
    return data[0][0];
  }
}
// --Pinakas library: include definitions-----------------------------------------
#include "Pinakas.cpp"
#endif
