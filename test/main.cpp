#include "../include/Pinakas.hpp"
#include <iostream>



int main()
{
  puts("\033[H\033[J");
  Pinakas::Matrix m = {{1, 2, 3}, 
                       {4, 5, 6},
                       {7, 8, 9}};

  Pinakas::Matrix result = round((m + 1/m)*10);

  std::cout << "dimensions:\n" << result.size << '\n';
  std::cout << "values:\n" << result << '\n';

  /*

  //*/
}