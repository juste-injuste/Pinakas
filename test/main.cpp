#include "../include/Pinakas.hpp"
#include <iostream>

int main()
{
  using namespace Pinakas;
  Matrix x(3, 3);
  Matrix y = {1, 2, 3};
  std::cout << y;

  std::cin.get();
}