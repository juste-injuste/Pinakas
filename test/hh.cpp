#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void qr_decomp(vector<vector<double>>& A, vector<vector<double>>& Q, vector<vector<double>>& R) {
    int m = A.size();
    int n = A[0].size();

    Q.resize(m, vector<double>(m, 0.0));
    R.resize(m, vector<double>(n, 0.0));

    for (int k = 0; k < n; k++) {
        vector<double> x(m - k, 0.0);

        for (int i = k; i < m; i++) {
            x[i - k] = A[i][k];
        }

        double norm_x = 0.0;

        for (int i = 0; i < m - k; i++) {
            norm_x += x[i] * x[i];
        }

        double sign = (x[0] >= 0) ? 1.0 : -1.0;
        double alpha = sign * sqrt(norm_x);

        vector<double> u(m - k, 0.0);

        for (int i = 0; i < m - k; i++) {
            u[i] = x[i];
        }

        u[0] += alpha;
        double norm_u = 0.0;

        for (int i = 0; i < m - k; i++) {
            norm_u += u[i] * u[i];
        }

        for (int i = 0; i < m - k; i++) {
            u[i] /= sqrt(norm_u);
        }

        for (int i = k; i < m; i++) {
            for (int j = k; j < n; j++) {
                R[i][j] += u[i - k] * A[i][j];
            }
        }

        for (int i = k; i < m; i++) {
            for (int j = 0; j < m - k; j++) {
                Q[i][j + k] -= 2 * u[i - k] * Q[i][j + k];
            }
        }

        for (int i = k; i < m; i++) {
            A[i][k] = alpha;

            for (int j = k + 1; j < n; j++) {
                A[i][j] -= 2 * u[i - k] * R[k][j];
            }
        }
    }
}

int main() {
    vector<vector<double>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    vector<vector<double>> Q;
    vector<vector<double>> R;

    qr_decomp(A, Q, R);

    cout << "A = " << endl;

    for (auto row : A) {
        for (auto element : row) {
            cout << element << " ";
        }

        cout << endl;
    }

    cout << "Q = " << endl;

    for (auto row : Q) {
        for (auto element : row) {
            cout << element << " ";
        }

        cout << endl;
    }

    cout << "R = " << endl;

    for (auto row : R) {
        for (auto element : row) {
            cout << element << " ";
        }

        cout << endl;
    }

    return 0;
}