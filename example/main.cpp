#define PINAKAS_LOGGING

#include "../src/Pinakas.cpp"
#include <iostream>

int main()
{
  using namespace Pinakas;

  // auto x_lo = linspace(0, 3, 15);
  // auto y_lo = (x_lo^2) + sin(x_lo*5) - x_lo/2;

  // auto y_hi = resample(y_lo, 10);
  // auto x_hi = linspace(x_lo(0), x_lo(-1), y_hi.numel());

  // plot({Set{"lo", x_lo, y_lo}, Set{"hi", x_hi, y_hi}});
  
  Matrix<double> A(2, 3, Random{0, 1});
  Matrix<double> B(2, 3, Random{0, 1});
  Matrix<float>  C(2, 3, Random{0, 1});
  A += B;
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
  A + 1.0;
  1.0f + A;
  1.0  + A;
  +A;
  A + Matrix<double>{2, 3, 0};
  A + Matrix<float>{2, 3, 0};
  Matrix<double>{2, 3, 0} + A;
  Matrix<float>{2, 3, 0}  + A;
  Matrix<double>{2, 3, 0} + Matrix<double>{2, 3, 0};
  Matrix<float>{2, 3, 0}  + Matrix<double>{2, 3, 0};
  Matrix<double>{2, 3, 0} + Matrix<float>{2, 3, 0};
  Matrix<double>{2, 3, 0} + Random{0, 1};
  Random{0, 1} + Matrix<double>{2, 3, 0};
  Matrix<double>{2, 3, 0} + 1.0;
  1.0 + Matrix<double>{2, 3, 0};
  Matrix<float>{2, 3, 0} + 1.0;
  1.0 + Matrix<float>{2, 3, 0};
  +Matrix<double>(2, 3, 0);
}