#include <complex>
#include <iostream>
#include <valarray>

const double PI = 3.141592653589793238460;

typedef std::complex<double> complex;
typedef std::valarray<complex> CArray;

void fft(CArray &x)
{
    // DFT
    unsigned int N = x.size(); // Size of the input array
    unsigned int k = N; // Current stage size
    unsigned int n; // Size of butterfly operations
    double thetaT = 3.14159265358979323846264338328L / N; // Angle for twiddle factor
    complex phiT = complex(cos(thetaT), -sin(thetaT)); // Twiddle factor for the first stage
    complex T; // Twiddle factor for each butterfly operation

    // Perform the FFT computation
    while (k > 1) {
        n = k;
        k >>= 1; // Halve the stage size
        phiT = phiT * phiT; // Square the twiddle factor for the next stage
        T = 1.0L; // Initialize the twiddle factor for the current stage

        // Perform butterfly operations
        for (unsigned int l = 0; l < k; l++)
        {
            for (unsigned int a = l; a < N; a += n)
            {
                unsigned int b = a + k; // Index of the element to combine with 'a'
                complex t = x[a] - x[b]; // Difference between 'a' and 'b'
                x[a] += x[b]; // Sum of 'a' and 'b'
                x[b] = t * T; // Multiply 't' by the twiddle factor
            }
            T *= phiT; // Update the twiddle factor for the next butterfly operation
        }
    }

    // Decimate the results
    unsigned int m = (unsigned int)log2(N); // Number of bits required to represent 'N'
    for (unsigned int a = 0; a < N; a++)
    {
        unsigned int b = a;

        // Reverse the bits of 'a'
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);

        if (b > a)
        {
            // Swap elements at indices 'a' and 'b'
            complex t = x[a];
            x[a] = x[b];
            x[b] = t;
        }
    }

    //// Normalize the results (commented out)
    //Complex f = 1.0 / sqrt(N); // Scaling factor for normalization
    //for (unsigned int i = 0; i < N; i++)
    //    x[i] *= f; // Scale each element by the factor 'f'
}


// inverse fft (in-place)
void ifft(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft( x );

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();
}

int main()
{
    const complex test[] = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    CArray data(test, 8);

    // forward fft
    fft(data);

    std::cout << "fft" << std::endl;
    for (int i = 0; i < 8; ++i)
    {
        std::cout << data[i] << std::endl;
    }

    // inverse fft
    ifft(data);

    std::cout << std::endl << "ifft" << std::endl;
    for (int i = 0; i < 8; ++i)
    {
        std::cout << data[i] << std::endl;
    }
    return 0;
}