#include "../include/Pinakas.hpp"
#include <iostream>

int main()
{
  using namespace Pinakas;
  puts("\033[H\033[J");

  Matrix x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  puts("start");
  x + x;
  puts("end");

  //Matrix x_work = {x^2, x^1, x^0};
  //auto QR = MGS(x_work);
  /*
  //std::cout << "x_work:\n" << x_work;
  std::cout << "Q:\n" << QR[0];
  std::cout << "R:\n" << QR[1];
  std::cout << "QxR:\n" << mul(QR[0], QR[1]);
  puts("\033[H\033[J");
  puts("start from here !!!!!!!!!!!!!!!!!");
  std::cout << "diff:\n" << sum(mul(QR[0], QR[1]) - x_work) << '\n';
  
  */

  //auto QR = FULL_MGS(x_work);
  //std::cout << "QxR: \n" << mul(QR[0], QR[1]);
}