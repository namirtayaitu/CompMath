import numpy as np


def jacobi_method(A, b, x0=None, epsilon=0.000001, max_iterations=100):
    """
    Solves Ax = b using Jacobi Iteration Method.
    All output uses standard decimal format (no scientific notation).
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    # Check for zero on diagonal
    for i in range(n):
        if abs(A[i, i]) < 1e-12:
            raise ValueError(f"Diagonal element A[{i},{i}] is zero.")

    print(f"{'Iter':<5} {'x1':<12} {'x2':<12} {'x3':<12} {'x4':<12} {'Max |Δx|':<12}")
    print("-" * 70)

    x_old = x.copy()
    for iteration in range(max_iterations):
        x_new = np.zeros(n)

        for i in range(n):
            s = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        # Compute max change
        delta = np.abs(x_new - x_old)
        max_delta = np.max(delta)

        # Print current iterate
        print(
            f"{iteration:<5} "
            f"{x_new[0]:<12.8f} "
            f"{x_new[1]:<12.8f} "
            f"{x_new[2]:<12.8f} "
            f"{x_new[3]:<12.8f} "
            f"{max_delta:<12.8f}"
        )

        if max_delta < epsilon:
            return x_new, iteration + 1

        x_old = x_new.copy()

    raise RuntimeError(f"Did not converge after {max_iterations} iterations")


# Define original system
A_orig = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
])
b_orig = np.array([18, 26, 34, 82])

# Reorder for better convergence: [2, 3, 0, 1] → rows 3,4,1,2 (0-indexed)
order = [2, 3, 0, 1]
A = A_orig[order]
b = b_orig[order]

print("Solving using Jacobi Iteration Method")
print("Matrix reordered for improved diagonal dominance.\n")

print("Reordered system:")
for i in range(4):
    eq = " + ".join([
        f"{A[i, j]:+g}*x{j + 1}" if A[i, j] >= 0 else f"{A[i, j]:g}*x{j + 1}"
        for j in range(4)
    ])
    print(f"{eq} = {b[i]}")
print()

try:
    solution, iters = jacobi_method(A, b, x0=[0, 0, 0, 0], epsilon=0.000001, max_iterations=100)

    print(f"\n✅ Converged in {iters} iterations")
    print("Solution (in reordered variable order):")
    for i in range(4):
        print(f"x{i + 1} = {solution[i]:.8f}")

    # Verify against ORIGINAL system (important!)
    residual = A_orig @ solution - b_orig
    print("\nVerification against ORIGINAL system:")
    print("A_original @ x =", " ".join(f"{val:12.8f}" for val in (A_orig @ solution)))
    print("Target b         =", " ".join(f"{val:12.8f}" for val in b_orig))
    print("Residual         =", " ".join(f"{val:12.8f}" for val in residual))
    print(f"Max |residual|   = {np.max(np.abs(residual)):.8f}")

except Exception as e:
    print("Error:", e)