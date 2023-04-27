#include "../include/matrix.hpp"
#include "../include/mudelizer.hpp"
#include <iostream>

int main()
{
    matrix m(3, 3);
    m(0, 0) = 1;
    m(1, 0) = 1;
    m(2, 0) = 2;
    m(0, 1) = 4;
    m(1, 1) = 5;
    m(2, 1) = 6;
    m(0, 2) = 7;
    m(1, 2) = 8;
    m(2, 2) = 9;
    m.show();

    auto QR = MGS(m);
    QR[0].show();
    QR[1].show();

}