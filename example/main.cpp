#include "../src/Pinakas.cpp"
#include <iostream>

int main()
{
  using namespace Pinakas;
  
  auto m = Matrix<float>(3, 3, Random(0, 10));
  auto s = m(Range{0, 1}, Range{0, 1});

  m = s;

  std::cout << m;
  std::cout << floor(m);
}