#include <iostream>
#include <immintrin.h> // Include AVX-512 intrinsics

float ComputeEuclideanSquaredNorm(float* data, int size)
{
    __m512 sum = _mm512_setzero_ps(); // Initialize a vector to store the sum

    // Process 16 elements at a time (assuming float is 32 bits)
    for (int i = 0; i < size; i += 16)
    {
        // Load 16 elements into a SIMD register
        __m512 values = _mm512_loadu_ps(data + i);

        // Compute the squared values using SIMD instructions
        __m512 squaredValues = _mm512_mul_ps(values, values);

        // Accumulate the squared values to the sum vector
        sum = _mm512_add_ps(sum, squaredValues);
    }

    // Compute the final sum by horizontally adding the elements in the sum vector
    __m256 sum256 = _mm512_extractf32x8_ps(sum, 0);
    __m256 sum128 = _mm256_add_ps(sum256, _mm512_extractf32x8_ps(sum, 1));
    sum128 = _mm256_hadd_ps(sum128, sum128);
    sum128 = _mm256_hadd_ps(sum128, sum128);

    // Extract the result from the sum vector
    float result;
    _mm_store_ss(&result, _mm256_castps256_ps128(sum128));

    return result;
}

int main()
{
    float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };

    float squaredNorm = ComputeEuclideanSquaredNorm(data, sizeof(data) / sizeof(float));
    std::cout << "Squared norm: " << squaredNorm << std::endl;

    return 0;
}
