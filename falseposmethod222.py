import numpy as np
import matplotlib.pyplot as plt


def false_position_method(f, a, b, epsilon, max_iterations):

    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        print("Error: f(a) and f(b) must have opposite signs.")
        return None, []

    print(f"{'Iter':<5} {'a':<14} {'b':<14} {'c':<14} {'f(c)':<14}")
    print("-" * 64)

    history = []
    for n in range(max_iterations):
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)

        history.append((n, a, b, c, fc))
        print(f"{n:<5} {a:<14.8f} {b:<14.8f} {c:<14.8f} {fc:<14.8f}")

        if abs(fc) < epsilon:
            return c, history

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    print(f"Failed to converge after {max_iterations} iterations")
    return None, history

def f(x):
    return x ** 3 - 2 * x - 5


if __name__ == "__main__":
    a = 2.0
    b = 3.0
    epsilon = 0.000001
    max_iter = 30

    print("Solving f(x) = x^3 - 2x - 5 = 0 using False Position Method")
    print(f"Initial interval: [{a}, {b}]")
    print(f"Tolerance (epsilon): {epsilon:.8f}\n")

    root, history = false_position_method(f, a, b, epsilon, max_iter)

    if root is not None:
        print(f"\n✅ Root found: {root:.8f}")
        print(f"f(root) = {f(root):.8f}")

        x_vals = np.linspace(1.5, 3.0, 400)
        y_vals = f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=r'$f(x) = x^3 - 2x - 5$', color='darkgreen')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(root, color='red', linestyle='--', label=f'Root ≈ {root:.6f}')
        plt.scatter([root], [f(root)], color='red', zorder=5, s=60)

        plt.scatter([a, b], [f(a), f(b)], color='blue', label='Initial endpoints')

        for i, (n, a_i, b_i, c_i, _) in enumerate(history[:5]):
            xa, xb = a_i, b_i
            ya, yb = f(xa), f(xb)
            plt.plot([xa, xb], [ya, yb], 'orange', linestyle='--', alpha=0.7)

        plt.xlim(1.5, 3.0)
        plt.ylim(-2, 8)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('False Position Method: Solving $x^3 - 2x - 5 = 0$')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()