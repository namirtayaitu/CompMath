import numpy as np
import matplotlib.pyplot as plt


def secant_method(f, x0, x1, epsilon, max_iterations):
    history = []
    x_prev = x0
    x_curr = x1
    f_prev = f(x_prev)
    f_curr = f(x_curr)

    print(f"{'Iter':<5} {'x_n':<14} {'f(x_n)':<14} {'|dx|':<14}")
    print("-" * 52)

    print(f"Init0 {x_prev:<14.8f} {f_prev:<14.8f} {'-':<14}")
    print(f"Init1 {x_curr:<14.8f} {f_curr:<14.8f} {'-':<14}")

    for n in range(max_iterations):
        if abs(f_curr - f_prev) < 1e-12:
            print("Denominator too small (f(x_n) ≈ f(x_{n-1})). Stopping.")
            return None, history

        dx = f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        x_next = x_curr - dx
        error = abs(dx)

        history.append((n, x_curr, f_curr, error))
        print(f"{n + 1:<5} {x_next:<14.8f} {f(x_next):<14.8f} {error:<14.8f}")

        if error < epsilon:
            return x_next, history

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f(x_next)

    print(f"Failed to converge after {max_iterations} iterations")
    return None, history

def f(x):
    return x ** 3 - 2 * x - 5

if __name__ == "__main__":
    x0 = 1.0  # First guess
    x1 = 3.0  # Second guess
    epsilon = 0.000001
    max_iter = 20

    print("Solving x^3 - 2x - 5 = 0 using Secant Method")
    print(f"Initial guesses: x0 = {x0}, x1 = {x1}")
    print(f"Epsilon (tolerance): {epsilon:.8f}\n")

    root, history = secant_method(f, x0, x1, epsilon, max_iter)

    if root is not None:
        print(f"\n✅ Root found: {root:.8f}")
        print(f"f(root) = {f(root):.8f}")

        # --- Plotting ---
        x_vals = np.linspace(1.0, 3.0, 400)
        y_vals = f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=r'$f(x) = x^3 - 2x - 5$', color='purple')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(root, color='red', linestyle='--', label=f'Root ≈ {root:.6f}')
        plt.scatter([root], [f(root)], color='red', zorder=5, s=60)

        # Plot secant lines for first few iterations
        if history:
            points = [x0, x1]
            for entry in history[:4]:  # First 4 updates
                points.append(entry[1])  # x_curr before update
            # Now plot secant lines between consecutive pairs
            for i in range(len(points) - 1):
                x_a, x_b = points[i], points[i + 1]
                y_a, y_b = f(x_a), f(x_b)
                plt.plot([x_a, x_b], [y_a, y_b], 'orange', linestyle='--', alpha=0.7)
                plt.scatter([x_a, x_b], [y_a, y_b], color='orange', s=30)

        plt.xlim(1.0, 3.0)
        plt.ylim(-6, 16)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Secant Method: Solving $x^3 - 2x - 5 = 0$')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()