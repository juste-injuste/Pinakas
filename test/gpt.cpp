#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void GPT(vector<vector<float>>& A) {
    int M = A.size();
    int N = A[0].size();
    vector<vector<float>> Q(M, vector<float>(N, 0.0));
    vector<vector<float>> R(N, vector<float>(N, 0.0));
    
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            Q[i][j] = A[i][j];
        }
        for (int k = 0; k < j; k++) {
            float dot = 0.0;
            for (int i = 0; i < M; i++) {
                dot += Q[i][k] * A[i][j];
            }
            for (int i = 0; i < M; i++) {
                Q[i][j] -= dot * Q[i][k];
            }
        }
        float norm = 0.0;
        for (int i = 0; i < M; i++) {
            norm += Q[i][j] * Q[i][j];
        }
        norm = sqrt(norm);
        for (int i = 0; i < M; i++) {
            Q[i][j] /= norm;
        }
        for (int i = j; i < N; i++) {
            float dot = 0.0;
            for (int k = 0; k < M; k++) {
                dot += Q[k][j] * A[k][i];
            }
            R[j][i] = dot;
        }
    }
    
    cout << "Q:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << Q[i][j] << " ";
        }
        cout << "\n";
    }
    
    cout << "R:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << R[i][j] << " ";
        }
        cout << "\n";
    }
}

int main() {
    int M, N;
    vector<vector<float>> A(M, vector<float>(N, 0.0));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cin >> A[i][j];
        }
    }
    modifiedGramSchmidt(A);
    return 0;
}