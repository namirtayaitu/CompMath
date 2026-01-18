import numpy as np
import matplotlib.pyplot as plt

def bisection_method(f, a, b, epsilon, max_iterations):

    n = 0
    fa = f(a)
    fb = f(b)
    history = []

    if fa * fb >= 0:
        print("Error: f(a) and f(b) must have opposite signs.")
        return None, history

    print(f"{'Iter':<5} {'a':<14} {'b':<14} {'c':<14} {'f(c)':<14} {'Width':<14}")
    print("-" * 76)

    while n <= max_iterations:
        c = (a + b) / 2.0
        fc = f(c)
        width = b - a

        history.append((n, a, b, c, fc, width))

        print(
            f"{n:<5} "
            f"{a:<14.8f} "
            f"{b:<14.8f} "
            f"{c:<14.8f} "
            f"{fc:<14.8f} "
            f"{width:<14.8f}"
        )

        if abs(fc) < epsilon or (b - a) / 2.0 < epsilon:
            return c, history

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        n += 1

    print(f"Failed to converge after {max_iterations} iterations")
    return None, history

if __name__ == "__main__":
    def f(x):
        return x**3 - 2*x - 5

    a = 1.0
    b = 3.0
    epsilon = 0.000001
    max_iter = 100

    print(f"Solving f(x) = x^3 - 2x - 5 = 0")
    print(f"Initial interval: [{a}, {b}]")
    print(f"Epsilon (tolerance): {epsilon:.8f}")
    print()

    result, history = bisection_method(f, a, b, epsilon, max_iter)

    if result is not None:
        print(f"\n✅ Root found: {result:.8f}")
        print(f"f(root) = {f(result):.8f}")

        # Plotting with NumPy
        x_vals = np.linspace(0, 4, 400)
        y_vals = f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=r'$f(x) = x^3 - 2x - 5$', color='steelblue')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(result, color='red', linestyle='--', label=f'Root ≈ {result:.6f}')
        plt.scatter([result], [f(result)], color='red', zorder=5, s=50)

        plt.axvline(a, color='green', linestyle=':', alpha=0.7, label='Initial interval bounds')
        plt.axvline(b, color='green', linestyle=':', alpha=0.7)

        plt.xlim(0, 4)
        plt.ylim(-10, 20)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Bisection Method Solution for $x^3 - 2x - 5 = 0$')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()