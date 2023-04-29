#include <iostream>
#include "matrix.h"
using namespace std;

int main()
{
    //QR factorization
    Matrix E = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
    E.print("E:");
    print("\nQR factorization of E:\n");
    Matrix Q, R;
    QR_factorization(E, &Q, &R);
    Q.print("Q:");
    R.print("R:");
    (Q * R).print("Check QR = B");

    return 0;
}