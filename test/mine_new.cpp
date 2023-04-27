#include "../include/Pinakas.hpp"
using namespace Pinakas::Backend;
Matrix* MGS_GPT(Matrix& A)
  {
    size_t i, j, k;
    size_t M = A.size.M;
    size_t N = A.size.N;
    Matrix V(A);
    Matrix Q(M, N, 0);
    Matrix R(N, N, 0);
    Value sum_of_squares, norm, projection;

    for (j = 0; j < N; ++j) {
      // q_j = v_j / ||v_j||_2
      for (i = 0, sum_of_squares = 0; i < M; ++i)
        sum_of_squares += V[i][j] * V[i][j];
      norm = std::sqrt(sum_of_squares);
      if (norm) for (i = 0; i < M; ++i)
        Q[i][j] += V[i][j] / norm;
      for (k = j; k < N; ++k) {
        // v_k = v_k - (qT_j*v_k)*q_j
        for (i = 0, projection = 0; i < M; ++i)
          projection += Q[i][j] * V[i][k];
        for (i = 0; i < M; ++i)
          V[i][k] -= projection * Q[i][j];
        // compute R
        if (k >= j) R[j][k] = projection;
      }
    }

    std::cout << "A:\n" << A;
    std::cout << "Q:\n" << Q;
    std::cout << "R:\n" << R;
    std::cout << "Q*R:\n" << mul(Q, R);

    Matrix* QR = new Matrix[2]{Q, R};
    return QR;
  }