#include "../src/Pinakas.cpp"

int main()
{
  using namespace Pinakas;

  Matrix<double> A(2, 3, 0);
  Matrix<double> B(2, 3, 1);
  Matrix<float>  C(2, 3, 2);

  std::cout << "should be -2"      << '\n' << (A - C);
  std::cout << "should be -1"      << '\n' << (A - B);
  std::cout << "should be [-1, 0]" << '\n' << (A - Random{0, 1});
  std::cout << "should be [0, 1]"  << '\n' << (Random{0, 1} - A);
  std::cout << "should be 1"       << '\n' << (C - 1.0f);
  std::cout << "should be 1"       << '\n' << (C - 1.0);
  std::cout << "should be -1"      << '\n' << (1.0f - C);
  std::cout << "should be -1"      << '\n' << (1.0  - C);
  std::cout << "should be -1"      << '\n' << (A - 1.0f);
  std::cout << "should be -1"      << '\n' << (A - 1.0);
  std::cout << "should be 1"       << '\n' << (1.0f - A);
  std::cout << "should be 1"       << '\n' << (1.0  - A);
  std::cout << "should be -0"      << '\n' << (-A);
  std::cout << "should be -1"      << '\n' << (A - Matrix<double>(2, 3, 1));
  std::cout << "should be -1"      << '\n' << (A - Matrix<float>(2, 3, 1));
  std::cout << "should be 1"       << '\n' << (Matrix<double>(2, 3, 1) - A);
  std::cout << "should be 1"       << '\n' << (Matrix<float>(2, 3, 1)  - A);
  std::cout << "should be -1"      << '\n' << (Matrix<double>(2, 3, 1) - Matrix<double>(2, 3, 2));
  std::cout << "should be -1"      << '\n' << (Matrix<float>(2, 3, 1)  - Matrix<double>(2, 3, 2));
  std::cout << "should be -1"      << '\n' << (Matrix<double>(2, 3, 1) - Matrix<float>(2, 3, 2));
  std::cout << "should be [0, 1]"  << '\n' << (Matrix<double>(2, 3, 1) - Random{0, 1});
  std::cout << "should be [-1, 0]" << '\n' << (Random{0, 1} - Matrix<double>(2, 3, 1));
  std::cout << "should be 0"       << '\n' << (Matrix<double>(2, 3, 1) - 1.0);
  std::cout << "should be 0"       << '\n' << (1.0 - Matrix<double>(2, 3, 1));
  std::cout << "should be 0"       << '\n' << (Matrix<float>(2, 3, 1) - 1.0);
  std::cout << "should be 0"       << '\n' << (1.0 - Matrix<float>(2, 3, 1));
  std::cout << "should be -1"      << '\n' << (-Matrix<double>(2, 3, 1));
  std::cout << "should be -1"      << '\n' << (A -= B);
  std::cout << "should be [-2, -1]"<< '\n' << (A -= Random{0, 1});
  std::cout << "should be [-3, -2]"<< '\n' << (A -= 1);
}