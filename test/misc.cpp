
  Matrix& Matrix::operator=(Matrix& B);
  inline Matrix& Matrix::operator=(Matrix&& B);
  
  Matrix& Matrix::operator=(Value B);

  Matrix operator+(Matrix& A, Matrix& B);
  inline Matrix operator+(Matrix& A, Matrix&& B);
  inline Matrix operator+(Matrix&& A, Matrix& B);
  inline Matrix operator+(Matrix&& A, Matrix&& B);

  Matrix operator+(Matrix& A, Value B);
  inline Matrix operator+(Matrix&& A, Value B);
  inline Matrix operator+(Value A, Matrix& B);
  inline Matrix operator+(Value A, Matrix&& B);
  
  Matrix operator-(Matrix& A, Matrix& B);
  inline Matrix operator-(Matrix& A, Matrix&& B);
  inline Matrix operator-(Matrix&& A, Matrix& B);
  inline Matrix operator-(Matrix&& A, Matrix&& B);
  
  Matrix operator-(Matrix& A, Value B);
  inline Matrix operator-(Matrix&& A, Value B);
  inline Matrix operator-(Value A, Matrix& B);
  inline Matrix operator-(Value A, Matrix&& B);
  
  Matrix operator*(Matrix& A, Matrix& B);
  inline Matrix operator*(Matrix& A, Matrix&& B);
  inline Matrix operator*(Matrix&& A, Matrix& B);
  inline Matrix operator*(Matrix&& A, Matrix&& B);
  
  Matrix operator*(Matrix& A, Value B);
  inline Matrix operator*(Matrix&& A, Value B);
  inline Matrix operator*(Value A, Matrix& B);
  inline Matrix operator*(Value A, Matrix&& B);
  
  Matrix operator/(Matrix& A, Matrix& B);
  inline Matrix operator/(Matrix& A, Matrix&& B);
  inline Matrix operator/(Matrix&& A, Matrix& B);
  inline Matrix operator/(Matrix&& A, Matrix&& B);
  
  Matrix operator/(Matrix& A, Value B);
  inline Matrix operator/(Matrix&& A, Value B);
  inline Matrix operator/(Value A, Matrix& B);
  inline Matrix operator/(Value A, Matrix&& B);

  Matrix floor(Matrix& matrix);
  inline Matrix floor(Matrix&& matrix);
  
  Matrix round(Matrix& matrix);
  inline Matrix round(Matrix&& matrix);
  
  Matrix ceil(Matrix& matrix);
  inline Matrix ceil(Matrix&& matrix);