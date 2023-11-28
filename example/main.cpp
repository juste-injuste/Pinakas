#define PINAKAS_LOGGING

#include "../src/Pinakas.cpp"
#include <iostream>

int main()
{
  using namespace Pinakas;

  auto x_lo = linspace(0, 1, 10);
  auto y_lo = (x_lo^2) + sin(x_lo*5)/5 - 1;

  auto y_hi = resample(y_lo, 10);
  auto x_hi = linspace(0, 1, y_hi.numel());

  plot({"lo", "hi"},  {DataSet{x_lo, y_lo}, DataSet{x_hi, y_hi}});
}