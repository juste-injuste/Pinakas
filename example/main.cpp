#define NDEBUG
#include "../src/Pinakas.cpp"
#include <iostream>

int main()
{
  using namespace Pinakas;

  auto x_lo = linspace(0, 3, 15);
  auto y_lo = (x_lo^2) + sin(x_lo*5) - x_lo/2;

  auto y_hi = resample(y_lo, 10);
  auto x_hi = linspace(x_lo(0), x_lo(-1), y_hi.numel());

  plot({Set("lo", x_lo, y_lo), Set("hi", x_hi, y_hi)});

  CHRONOMETRO_MEASURE(5)
  {
    CHRONOMETRO_MEASURE(1000, nullptr, "iteration took: %ms")
    resample(y_lo, 1000);
  }
}