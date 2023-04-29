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
#define tic	auto _start = std::chrono::high_resolution_clock::now()	
#define toc	auto _stop = std::chrono::high_resolution_clock::now(); std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count() << " us\n"
// --Pinakas library: backend forward declaration---------------------------------
namespace Pinakas::Backend
{
  template<typename T>
  using List = std::initializer_list<T>;
  //
  typedef double Value;
  typedef std::function<Value(Value)> Function;
  //
  struct Size;
  struct Matrix;
// -------------------------------------------------------------------------------
  Matrix& operator+=(Matrix& A, const Matrix& B);
  Matrix& operator+=(Matrix& A, const Value B);
// -------------------------------------------------------------------------------
  Matrix operator+(const Matrix& A, const Matrix& B);
  inline Matrix&& operator+(Matrix&& A, Matrix&& B);
  Matrix operator+(const Matrix& A, const Value B);
  inline Matrix&& operator+(Matrix&& A, const Value B);
  inline Matrix operator+(const Value A, const Matrix& B);
  inline Matrix&& operator+(const Value A, Matrix&& B);
  //Matrix diff(const Matrix& A, size_t n = 1);
  //inline Matrix diff(const Matrix&& A, size_t n = 1);

  //Value newton(const Function function, const Value tolerance = 1e-6, const size_t max_iteration = 100, const Value seed = 1);
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
    bool operator==(const Size B) const noexcept;
    bool operator!=(const Size B) const noexcept;
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
      Matrix(Matrix&& matrix) noexcept;
      // create a matrix MxN
      Matrix(const size_t M, const size_t N);
      // create a matrix with the same dimensions as 'matrix'
      Matrix(const Size size);
      // create a matrix MxN with a specific value
      Matrix(const size_t M, const size_t N, const Value value);
      // create a matrix with the same dimensions as 'matrix' with a specific value
      Matrix(const Size size, Value value);
      // create a matrix with the same dimensions as 'matrix' with a specific value
      Matrix(const Matrix& matrix, Value value);
      // create a matrix MxN random values
      Matrix(const size_t M, const size_t N, const Value min, const Value max);
      // create a matrix with the same dimensions as 'matrix' with random values
      Matrix(const Size size, const Value min, const Value max);
      // create a matrix with the same dimensions as 'matrix' with random values
      Matrix(const Matrix& matrix, const Value min, const Value max);
      // create a matrix from specific values
      Matrix(const List<Value> values);
      // create a matrix from specific values
      Matrix(const List<const List<const Value>> values);
      // join matrix (side-wise)
      Matrix(const List<const Matrix> list);
      //
      Matrix& operator=(const Matrix& B) noexcept;
      Matrix& operator=(Matrix&& B) noexcept;
      Matrix& operator=(const Value B) noexcept;
      // vector indexing
      inline Value* operator[](const size_t y) const noexcept;
      // bound-checked flat-indexing
      inline Value& operator()(const size_t index) const noexcept;
      // bound-checked indexing
      inline Value& operator()(const size_t y, const size_t x) const noexcept;
      //
      inline Size size(void) const noexcept;
    private:
      // allocate memory block
      bool allocate(const size_t M, const size_t N) noexcept;
      // information regarding matrix size
      Size _size;
      // memory block for data
      std::unique_ptr<char[]> _memory_block;
      // data is a Value[M][N] array
      Value** _data;
  };
}
// --Pinakas library: operator overloads forward declarations----------------------
namespace Pinakas::Backend
{
  std::ostream& operator<<(std::ostream& ostream, const Matrix& matrix);
  std::ostream& operator<<(std::ostream& ostream, const Size size);
}
// --Pinakas library: backend struct and class member definitions-----------------
namespace Pinakas::Backend
{
  bool Size::operator==(const Size B) const noexcept {
    return (M == B.M) && (N == B.N) && (numel == B.numel);
  }

  bool Size::operator!=(const Size B) const noexcept {
    return (M != B.M) || (N != B.N) || (numel != B.numel);
  }

  bool Matrix::allocate(const size_t M, const size_t N) noexcept
  {
    // allocate memory
    _memory_block.reset(new char[sizeof(Value*[M]) + sizeof(Value[M][N])]);

    // get address of memory block
    char* address = _memory_block.get();

    // validate memory allocation
    if (!address) {
      std::cerr << "! invalid memory allocation !\n";
      return false;
    }

    // create rows into memory block
    _data = (Value**) address;
    // offset address
    address += sizeof(Value*) * M;
    
    for (size_t y = 0; y < M; ++y) {
      // create columns into memory block
      _data[y] = (Value*) address;
      // offset address
      address += sizeof(Value) * N;
    }

    // save size
    _size = {M, N, M*N};

    return true;
  }
  
  Matrix::~Matrix()
  {
    std::clog << "Matrix deleted !\n";
  }

  Matrix::Matrix()
    : // member initialization list
    _size{0, 0, 0},
    _memory_block(nullptr),
    _data(nullptr)
  {
    std::clog << "Matrix created ! (empty)\n";
  }

  Matrix::Matrix(const Matrix& matrix)
  {
    std::clog << "Matrix copied !\n";
    
    // allocate memory
    if (allocate(matrix._size.M, matrix._size.N)) {
      // assign value to matrix
      for (size_t index = 0; index < _size.numel; ++index)
        _data[0][index] = matrix[0][index];
    }
  }

  Matrix::Matrix(Matrix&& matrix) noexcept
    : // member initialization list
    _size(matrix._size),
    _memory_block(matrix._memory_block.release()),
    _data(matrix._data)
  {
    std::clog << "Matrix moved !\n";
  }

  Matrix::Matrix(const size_t M, const size_t N)
  {
    std::clog << "Matrix created !\n";
    
    // allocate memory
    allocate(M, N);
  }

  Matrix::Matrix(const Size size)
    : Matrix(size.M, size.N)
  {}

  Matrix::Matrix(const size_t M, const size_t N, const Value value)
  {
    std::clog << "Matrix created !\n";
    
    // allocate memory
    if (allocate(M, N)) {
      // assign value to matrix
      for (size_t index = 0; index < _size.numel; ++index)
        _data[0][index] = value;
    }
  }
  
  Matrix::Matrix(const Matrix& matrix, Value value)
    : Matrix(matrix._size.M, matrix._size.N, value)
  {}
  
  Matrix::Matrix(const Size size, Value value)
    : Matrix(size.M, size.N, value)
  {}

  Matrix::Matrix(const size_t M, const size_t N, const Value min, const Value max)
  {
    std::clog << "Matrix created !\n";

    // allocate memory
    if (allocate(M, N)) {
      // random number generator
      std::random_device device;
      std::mt19937 generator(device());
      std::uniform_real_distribution<> uniform_distribution(min, max);

      // assign random value to matrix
      for (size_t index = 0; index < _size.numel; ++index)
        _data[0][index] = uniform_distribution(generator);
    }
  }
  
  Matrix::Matrix(const Matrix& matrix, Value min, Value max)
    : Matrix(matrix._size.M, matrix._size.N, min, max)
  {}
  
  Matrix::Matrix(const Size size, Value min, Value max)
    : Matrix(size.M, size.N, min, max)
  {}


  Matrix::Matrix(const List<Value> list)
  {
    std::clog << "Matrix created !\n";
    
    // allocate memory
    if (allocate(1, list.size())) {
      // assign values into matrix
      size_t x = 0;
      for (Value value : list)
        _data[0][x++] = value;
    }
  }

  Matrix::Matrix(const List<const List<const Value>> values)
  {
    std::clog << "Matrix created !\n";

    // dimension validation
    size_t temp_N = 0;
    for (List<const Value> vector : values) {
      if (temp_N && (temp_N != vector.size())) {
        std::cerr << "vertical dimensions mismatch (" << temp_N << " vs " << vector.size() << ")\n";
        _size = {0, 0, 0};
        return;
      }
      else temp_N = vector.size();
    }
    
    // allocate memory
    if (allocate(values.size(), temp_N)) {
      // assign values into matrix
      size_t y = 0;
      for (List<const Value> vector : values) {
        size_t x = 0;
        for (Value value : vector) {
          _data[y][x] = value;
          ++x;
        }
        ++y;
      }
    }
  }
  
  Matrix::Matrix(const List<const Matrix> list)
  {
    std::clog << "Matrix created !\n";

    // dimension validation
    size_t temp_M = 0;
    size_t temp_N = 0;
    for (Matrix matrix : list) {
      if (temp_M && (temp_M != matrix._size.M)) {
        std::cerr << "vertical dimensions mismatch (" << temp_M << " vs " << matrix._size.M << ")\n";
        _size = {0, 0, 0};
        return;
      }
      else temp_M = matrix._size.M;
      temp_N += matrix._size.N;
    }

    // allocate memory
    if (allocate(temp_M, temp_N)) {
      size_t index = 0;
      for (Matrix matrix : list) {
        for (size_t y = 0; y < matrix._size.M; ++y)
          for (size_t x = 0; x < matrix._size.N; ++x)
            _data[y][x + index] = matrix[y][x];
        index += matrix._size.N;
      }
    }
  }

  Value min(const Matrix& matrix)
  {
    Value minimum = matrix[0][0];

    // find minimal value
    for (size_t index = 1; index < matrix.size().numel; ++index)
      if (matrix[0][index] < minimum) minimum = matrix[0][index];

    return minimum;
  }

  Value min(const Matrix&& matrix)
  {
    return min(const_cast<Matrix&>(matrix));
  }

  Value max(const Matrix& matrix)
  {
    Value maximum = matrix[0][0];

    // find minimal value
    for (size_t index = 1; index < matrix.size().numel; ++index)
      if (matrix[0][index] > maximum) maximum = matrix[0][index];
      
    return maximum;
  }

  Value max(const Matrix&& matrix)
  {
    return max(const_cast<Matrix&>(matrix));
  }

  Value sum(const Matrix& matrix)
  {
    Value summation = 0;

    for (size_t index = 0; index < matrix.size().numel; ++index)
      summation += matrix[0][index];
    puts("sum end");
    return summation;
  }

  Value sum(const Matrix&& matrix)
  {
    return sum(const_cast<Matrix&>(matrix));
  }

  inline Value* Matrix::operator[](const size_t index) const noexcept
  {
    return _data[index];
  }

  inline Value& Matrix::operator()(const size_t index) const noexcept
  {
    // return value if the index is valid
    if (index < _size.N*_size.M) return _data[0][index];
    // out of bound error message
    std::cerr << '(' << index << ") out of bound " << _size.N*_size.M << " (dimensions are " << _size << ")\n";
    return _data[0][0];
  }

  inline Value& Matrix::operator()(const size_t y, const size_t x) const noexcept
  {
    // return value if the index is valid
    if ((y < _size.M) && (x < _size.N)) return _data[y][x];
    // out of bound error messages
    else if (y >= _size.M)
      std::cerr << '(' << y << ",_) out of bound " << _size.M << " (dimensions are " << _size << ")\n";
    else if (x >= _size.N)
      std::cerr << "(_," << x << ") out of bound " << _size.N << " (dimensions are " << _size << ")\n";
    return _data[0][0];
  }

  
  inline Size Matrix::size(void) const noexcept
  {
    return _size;
  }
}
#endif
