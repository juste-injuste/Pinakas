# THIS DOESNT WORK BUT IT IS CLOSE ??

import numpy as np

def levinson_durbin(V):
    # Initialize the first two coefficients
    P = -V[1] / V[0]
    K = V[0] + P * V[1]
    A = [1, P]

    for m in range(2, len(V)):
        # Calculate reflection coefficient
        P = -np.dot(A[-1::-1], V[1:m+1]) / K

        # Update K
        K *= 1 - P**2

        # Update the LPC coefficients
        A.append(P)
        for i in range(1, m//2 + 1):
            A[i], A[m-i] = A[i] + P * A[m-i], A[m-i] + P * A[i]

    return np.array(A)

# Example usage:
V = np.array([1, 0.5, 0.3, 0.2])
M = np.array([[1, 0.5, 0.3, 0.2],
              [0.5, 1, 0.5, 0.3],
              [0.3, 0.5, 1, 0.5],
              [0.2, 0.3, 0.5, 1]])

# Solve for LPC coefficients using Levinson-Durbin
LPC = levinson_durbin(V)

print("Autocorrelation Vector V:")
print(V)
print("\nAutocorrelation Matrix M:")
print(M)
print("\nLPC Coefficients:")
print(LPC)
