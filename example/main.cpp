// #define PINAKAS_LOGGING

#include "../src/Pinakas.cpp"
#include "Nimata.hpp"
#include <iostream>

int main()
{
  using namespace Pinakas;
  using namespace Parallilos;

  size_t M = 10;
  size_t N = 10;
  Matrix<float> aM(M, N, {0, 1});
  Matrix<float> bM(M, N, {0, 1});
  Matrix<float> cM(M, N);

  float* a = aM.data();
  float* b = bM.data();
  float* c = cM.data();

  for (const size_t k : SIMD<float>::parallel(aM.size().numel))
  {
    simd_storea(c+k, simd_add(simd_loada(a+k), simd_loada(b+k)));
  }

  for (const size_t k : SIMD<float>::sequential(aM.size().numel))
  {
    c[k] = a[k] + b[k];
  }

  std::cout << "c:\n" << cM;
  std::cout << "error:\n" << (cM - (aM + bM));
  std::cout << SIMD<float>::set << '\n';

}