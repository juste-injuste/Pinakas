#include <complex>
#include <cmath>
#include <vector>
#include <iostream>
#define M_PI 3.14159265358979323846


typedef std::complex<double> Complex;

std::vector<Complex> fft(const std::vector<Complex>& values) {
    const int N = values.size();

    if (N <= 1)
        return values;

    std::vector<Complex> even(N / 2);
    std::vector<Complex> odd(N / 2);
    for (int i = 0; i < N / 2; i++) {
        even[i] = values[i * 2];
        odd[i] = values[i * 2 + 1];
    }

    std::vector<Complex> transformed_even = fft(even);
    std::vector<Complex> transformed_odd = fft(odd);

    std::vector<Complex> transformed_values(N);
    for (int k = 0; k < N / 2; k++) {
        Complex t = std::polar(1.0, -2 * M_PI * k / N) * transformed_odd[k];
        transformed_values[k] = transformed_even[k] + t;
        transformed_values[k + N / 2] = transformed_even[k] - t;
    }

    return transformed_values;
}

std::vector<Complex> ifft(const std::vector<Complex>& values) {
    const int N = values.size();

    if (N <= 1)
        return values;

    std::vector<Complex> even(N / 2);
    std::vector<Complex> odd(N / 2);
    for (int i = 0; i < N / 2; i++) {
        even[i] = values[i * 2];
        odd[i] = values[i * 2 + 1];
    }

    std::vector<Complex> transformed_even = ifft(even);
    std::vector<Complex> transformed_odd = ifft(odd);

    std::vector<Complex> transformed_values(N);
    for (int k = 0; k < N / 2; k++) {
        Complex t = std::polar(1.0, 2 * M_PI * k / N) * transformed_odd[k];
        transformed_values[k] = transformed_even[k] + t;
        transformed_values[k + N / 2] = transformed_even[k] - t;
    }

    // Scale the values by dividing by N
    for (int i = 0; i < N; i++) {
        transformed_values[i] /= N;
    }

    return transformed_values;
}

// Example usage
int main() {
    std::vector<Complex> values = {Complex(1, 0), Complex(2, 0), Complex(3, 0), Complex(4, 0)};

    std::vector<Complex> transformed_values = fft(values);

    for (const Complex& value : transformed_values) {
        std::cout << value << " ";
    }
    
    transformed_values = ifft(transformed_values);

    for (const Complex& value : transformed_values) {
        std::cout << value << " ";
    }

    return 0;
}
