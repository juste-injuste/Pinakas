# Matrix Division
```text
x = div(b, A)
```

This function performs matrix division.

```cpp
template<typename MatrixLike, typename T = typename MatrixLike::Type>
Matrix<double> div(const MatrixLike& b, Matrix<double> A);
```

## Usage

The function takes two parameters:

`b` column matrix corresponding to `b` in `Ax = b`
`A` matrix corresponding to `A` in `Ax = b`


representing the left-hand side of the equation.
The function returns a column matrix (Matrix<double>) x, which is the solution to the linear system Rx = Qt*b, where R is an upper triangular matrix obtained from QR decomposition, Q is the orthogonal matrix obtained from QR decomposition, and t denotes the transpose operation.

## Error Handling
The function performs some error checks and throws std::invalid_argument exceptions in the following cases:

If the vertical dimensions of b and A do not match, it throws an exception indicating the vertical dimensions mismatch.
If the horizontal dimension of b is not 1, it throws an exception indicating that b is not a column matrix.

## Algorithm
The function uses QR decomposition implemented using the modified Gram-Schmidt process and back-substitution.

First, QR decompose A so A = QR
* QRx = b
* Q^-1QRx = Q^-1b
* Rx = Q^-1b
* Rx = Qtb



to decompose matrix A into an orthogonal matrix Q and an upper triangular matrix R. Then, it solves the linear system Rx = Qt*b using back substitution to obtain the solution matrix x.

The algorithm proceeds as follows:

Verify the dimensions of b and A.
Initialize necessary matrices for the QR decomposition.
Perform QR decomposition using the modified Gram-Schmidt process.
Solve the linear system Rx = Qt*b using back substitution.

## Example
```cpp
#include "Pinakas.hpp"
#include <iostream>

int main()
{
  using namespace Pinakas;

  Matrix<double> xdata = linspace(0, 1, 10).transpose();

  Matrix<double> b = 2*(xdata^2) - 0.5*xdata + 1.5;
  Matrix<double> A = {xdata^2, xdata^1, xdata^0};

  // solve Ax = b   ->   x = b/A
  Matrix<double> x = div(b, A);

  std::cout << "x:\n" << x;
}
```
Console output:

```text
x:
  2
0.5
1.5
```


## License
This code is released under the MIT [License](../LICENSE).