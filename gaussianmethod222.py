import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    M = np.hstack((A.astype(float), b.reshape(-1, 1)))

    print("Initial augmented matrix [A | b]:")
    for row in M:
        print(" ".join(f"{val:12.8f}" for val in row))
    print("\n--- Forward Elimination ---")

    for col in range(n):
        max_row = np.argmax(np.abs(M[col:, col])) + col
        if abs(M[max_row, col]) < 1e-12:
            raise ValueError("Matrix is singular.")

        if max_row != col:
            M[[col, max_row]] = M[[max_row, col]]
            print(f"\nSwapped rows {col} and {max_row}")

        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, col:] -= factor * M[col, col:]

        print(f"\nAfter column {col}:")
        for row in M:
            print(" ".join(f"{val:12.8f}" for val in row))

    print("\n--- Back Substitution ---")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(M[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i, -1] - s) / M[i, i]
        print(f"x[{i}] = {x[i]:.8f}")

    return x

A = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
])

b = np.array([18, 26, 34, 82])

try:
    x = gaussian_elimination(A, b)

    print("\n✅ Final Solution:")
    for i in range(len(x)):
        print(f"x{i + 1} = {x[i]:.8f}")

    Ax = A @ x
    residual = Ax - b
    print("\nVerification (A @ x):")
    print("A @ x =", " ".join(f"{val:12.8f}" for val in Ax))
    print("Target b =", " ".join(f"{val:12.8f}" for val in b))
    print("Residual =", " ".join(f"{val:12.8f}" for val in residual))
    print(f"Max absolute residual = {np.max(np.abs(residual)):.8f}")

except ValueError as e:
    print("Error:", e)