#include "../src/Pinakas.cpp"

int main()
{
  using namespace Pinakas;
  
  Matrix<double> A(2, 3, Random{0, 1});
  Matrix<double> B(2, 3, Random{0, 1});
  Matrix<float>  C(2, 3, Random{0, 1});

  A += B;
  A += C;
  A += Random{0, 1};
  A += 1;
  A + C;
  A + B;
  A + Random{0, 1};
  Random{0, 1} + A;
  C + 1.0f;
  C + 1.0;
  1.0f + C;
  1.0  + C;
  A + 1.0f;
  1.0f + A;
  +A;
  A + Matrix<double>(2, 3, 0);
  Matrix<double>(2, 3, 0) + A;
  C + Matrix<double>(2, 3, 0);
  Matrix<double>(2, 3, 0) + C;
  Matrix<double>(2, 3, 0) + Matrix<double>(2, 3, 0);
  Matrix<float>(2, 3, 0)  + Matrix<double>(2, 3, 0);
  Matrix<double>(2, 3, 0) + Matrix<float>(2, 3, 0);
  Matrix<double>(2, 3, 0) + Random{0, 1};
  Random{0, 1} + Matrix<double>(2, 3, 0);
  Matrix<double>(2, 3, 0) + 1.0;
  1.0 + Matrix<double>(2, 3, 0);
  Matrix<float>(2, 3, 0) + 1.0;
  1.0 + Matrix<float>(2, 3, 0);
  +Matrix<double>(2, 3, 0);
}