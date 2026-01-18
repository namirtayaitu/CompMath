import numpy as np


def sor_method(A, b, omega, x0=None, epsilon=0.000001, max_iterations=100):

    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    for i in range(n):
        if abs(A[i, i]) < 1e-12:
            raise ValueError(f"Diagonal element A[{i},{i}] is zero.")

    print(f"{'Iter':<5} {'x1':<12} {'x2':<12} {'x3':<12} {'x4':<12} {'Max |Δx|':<12}")
    print("-" * 70)

    for iteration in range(max_iterations):
        x_old = x.copy()
        max_delta = 0.0

        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))  # NEW values (j < i)
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))  # OLD values (j > i)
            x_gs = (b[i] - s1 - s2) / A[i, i]  # Gauss-Seidel estimate

            # SOR update
            x_new_i = (1 - omega) * x_old[i] + omega * x_gs
            delta_i = abs(x_new_i - x_old[i])

            if delta_i > max_delta:
                max_delta = delta_i
            x[i] = x_new_i

        print(
            f"{iteration:<5} "
            f"{x[0]:<12.8f} "
            f"{x[1]:<12.8f} "
            f"{x[2]:<12.8f} "
            f"{x[3]:<12.8f} "
            f"{max_delta:<12.8f}"
        )

        if max_delta < epsilon:
            return x.copy(), iteration + 1

    raise RuntimeError(f"Did not converge after {max_iterations} iterations")

A_orig = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
])
b_orig = np.array([18, 26, 34, 82])

order = [2, 3, 0, 1]
A = A_orig[order]
b = b_orig[order]

omega = 1.25

print("Solving using Successive Over-Relaxation (SOR) Method")
print(f"Relaxation parameter ω = {omega:.2f}")
print("Matrix reordered for improved diagonal dominance.\n")

print("Reordered system:")
for i in range(4):
    terms = []
    for j in range(4):
        coeff = A[i, j]
        if coeff >= 0:
            terms.append(f"+{coeff:g}*x{j + 1}")
        else:
            terms.append(f"{coeff:g}*x{j + 1}")
    eq = " ".join(terms)
    print(f"{eq} = {b[i]}")
print()

try:
    solution, iters = sor_method(A, b, omega=omega, x0=[0, 0, 0, 0], epsilon=0.000001, max_iterations=100)

    print(f"\n✅ Converged in {iters} iterations")
    print("Solution (in reordered variable order):")
    for i in range(4):
        print(f"x{i + 1} = {solution[i]:.8f}")

    Ax = A_orig @ solution
    residual = Ax - b_orig
    print("\nVerification against ORIGINAL system:")
    print("A_original @ x =", " ".join(f"{val:12.8f}" for val in Ax))
    print("Target b         =", " ".join(f"{val:12.8f}" for val in b_orig))
    print("Residual         =", " ".join(f"{val:12.8f}" for val in residual))
    print(f"Max |residual|   = {np.max(np.abs(residual)):.8f}")

except Exception as e:
    print("Error:", e)