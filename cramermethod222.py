import numpy as np

def cramer_method(A, b):
    n = len(b)
    if A.shape != (n, n):
        return "Error: A must be a square matrix of size n x n"

    D = np.linalg.det(A)
    if abs(D) < 1e-12:
        return "Error: System is either singular or not compatible"

    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        D_i = np.linalg.det(A_i)
        x[i] = D_i / D

    return x

A = np.array([
    [ 3,  -5,  47,  20],
    [11,  16,  17,  10],
    [56,  22,  11, -18],
    [17,  66, -12,   7]
])

b = np.array([18, 26, 34, 82])

solution = cramer_method(A, b)

print("=== Solving 4x4 System using Cramer's Rule ===")
print("\nCoefficient Matrix A:")
print(A)
print("\nRight-hand side vector b:")
print(b)
print("\nSolution vector x:")
if isinstance(solution, str):
    print(solution)
else:
    print(solution)
    print("\nVerification (A @ x should ≈ b):")
    print("A @ x =", A @ solution)