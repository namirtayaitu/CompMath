import numpy as np


def gauss_jordan(A, b):
    n = len(b)

    M = np.hstack((A.astype(float), b.reshape(-1, 1)))

    print("Initial augmented matrix [A | b]:")
    for row in M:
        print(" ".join(f"{val:12.8f}" for val in row))
    print("\n--- Gauss-Jordan Elimination ---")

    for col in range(n):
        max_row = np.argmax(np.abs(M[col:, col])) + col
        if abs(M[max_row, col]) < 1e-12:
            raise ValueError("Matrix is singular.")

        if max_row != col:
            M[[col, max_row]] = M[[max_row, col]]
            print(f"\nSwapped rows {col} and {max_row}")

        pivot = M[col, col]
        M[col, :] /= pivot
        print(f"\nAfter normalizing row {col} (pivot = {pivot:.8f}):")
        for row in M:
            print(" ".join(f"{val:12.8f}" for val in row))

        for row in range(n):
            if row != col:
                factor = M[row, col]
                M[row, :] -= factor * M[col, :]
        print(f"\nAfter eliminating column {col}:")
        for row in M:
            print(" ".join(f"{val:12.8f}" for val in row))

    x = M[:, -1]
    return x

A = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
])

b = np.array([18, 26, 34, 82])

print("Solving 4x4 system using Gauss-Jordan Elimination\n")

try:
    x = gauss_jordan(A, b)

    print("\n✅ Final Solution (from RREF last column):")
    for i in range(len(x)):
        print(f"x{i + 1} = {x[i]:.8f}")

    # Verification
    Ax = A @ x
    residual = Ax - b
    print("\nVerification:")
    print("A @ x =", " ".join(f"{val:12.8f}" for val in Ax))
    print("Target b =", " ".join(f"{val:12.8f}" for val in b))
    print("Residual =", " ".join(f"{val:12.8f}" for val in residual))
    print(f"Max absolute residual = {np.max(np.abs(residual)):.8f}")

except ValueError as e:
    print("Error:", e)