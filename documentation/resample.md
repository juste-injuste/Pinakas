# Resampling
```text
data_resampled = resample(data, L, keep = 2, alpha = 3.5, tail = false)
```

This function performs signal resampling by a factor L.

```cpp
template<typename T>
Matrix<double> resample(const Matrix<T>& data, unsigned L, unsigned keep=2, double alpha=3.5, const bool tail=false);
```

## Usage

The function takes the following arguments:

* `data` is the data that will be resampled. It is resampled row-wised.
* `L` is the resampling factor.
* `reflect = 2` is the amount of data to reflect. The default value is recommanded.
* `alpha = 3.5` is the filter length factor. The default value is recommanded.
* `tail = false` flags to keep the signal tail after resampling. The default value is recommanded.

## Example

```cpp
#include "Pinakas.hpp"

int main()
{
  using namespace Pinakas;

  auto x_lo = linspace(0, 3, 15);
  auto y_lo = (x_lo^2) + sin(x_lo*5) - x_lo/2;

  auto y_hi = resample(y_lo, 10);
  auto x_hi = linspace(x_lo(0), x_lo(-1), y_hi.numel());

  plot({Set("lo", x_lo, y_lo), Set("hi", x_hi, y_hi)});
}
```

## Algorithm

The function resamples data using the following steps:

* Create a blackman-windowed sinc impulsion of length `2*floor(L*alpha) + 1` with a frequency of `1/L`.
* Extend `data` by adding `reflect` amount of reflected data at the start and end of each row.
* Upsample with zero-padding the extended `data` by a factor `L`.
* Interpolate the extended `data` using a cropped-convolution with the impulsion.

Reflecting the data at its edges increase the resampled data's quality at its edges. Passed 2 the benefit is essentially non-observable.

The resulting data is of size `data.M()` by `L*data.N() - (tail ? 1 : L)`.

## Errors

* `'data' must be atleast 4 element wide` if rows do not contain enough elements.

## Warnings

* `'L' must be at least 2, 2 used instead` when an invalid resampling factor is used.
* `'reflect' must be less than the width of 'data', 2 used instead` when an invalid amount of reflected data is used.
* `'alpha' must be at least 1/L, 3.5 used instead` when an invalid filter length factor is used.

## Performance

The impulsion is cached to help with pointless recalculations. Changing `L` or `alpha` between `resample` calls invalidates the cached impulsion.

When compiling in release mode, none of the warnings will be validated or emitted.

## Static assertions

* `T` must be convertible to the double type.

## License

This code is released under the MIT [License](../LICENSE).