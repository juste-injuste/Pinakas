#include "../src/Pinakas.cpp"
#include <iostream>
#include "../include/Chronometro.hpp"

int main()
{
  using namespace Pinakas;

  // auto x_lo = Pinakas::linspace(0, 3, 15);
  // auto y_lo = (x_lo^2) + Pinakas::sin(x_lo*5) - x_lo/2;

  // auto y_hi = Pinakas::resample(y_lo, 10);
  // auto x_hi = Pinakas::linspace(x_lo(0), x_lo(-1), y_hi.numel());

  // Pinakas::plot({Pinakas::Set("lo", x_lo, y_lo), Pinakas::Set("hi", x_hi, y_hi)});

  // std::cout << x_hi;

  auto data = Matrix<double>({1, 2, 3});

  auto LPC  = lpc(data, 2);

  std::cout << "coefficients:\n" << LPC;
}