#include "../include/Pinakas.hpp"
#include <iostream>



int main()
{
  puts("\033[H\033[J");
  Pinakas::Matrix m1 = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}};
  Pinakas::Matrix big = {m1, m1, m1};
  auto QR = MGS(big);

  std::cout << "QxR:\n" << mul(QR[0], QR[1]);
  std::cout << "diff:\n" << sum(big - mul(QR[0], QR[1]));

  Pinakas::Matrix m2 = {{2, 4, 3}, 
                        {1, 5, 7},
                        {3, 7, 2}};
                        
  Pinakas::Matrix m3 = {{2, 4},
                        {1, 5},
                        {3, 7}};

  //std::cout << "m1:\n" << m1;
  //std::cout << "m2:\n" << m2;
  //std::cout << "m2:\n" << m3;

 // GPT_MGS(m1);

}