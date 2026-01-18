import numpy as np
import matplotlib.pyplot as plt


def newton_raphson_method(f, df, x0, epsilon, max_iterations):

    history = []
    x = x0

    print(f"{'Iter':<5} {'x_n':<14} {'f(x_n)':<14} {'|dx|':<14}")
    print("-" * 52)

    for n in range(max_iterations):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 0.000000000001:
            print("Derivative too close to zero. Stopping.")
            return None, history

        dx = fx / dfx
        x_new = x - dx
        error = abs(dx)

        history.append((n, x, fx, error))
        print(f"{n:<5} {x:<14.8f} {fx:<14.8f} {error:<14.8f}")

        if error < epsilon:
            return x_new, history

        x = x_new

    print(f"Failed to converge after {max_iterations} iterations")
    return None, history

def f(x):
    return x ** 3 - 2 * x - 5

def df(x):
    return 3 * x ** 2 - 2

if __name__ == "__main__":
    x0 = 2.0  #initial guess
    epsilon = 0.000001
    max_iter = 20

    print("Solving x^3 - 2x - 5 = 0 using Newton-Raphson Method")
    print(f"Initial guess x0 = {x0}")
    print(f"Epsilon (tolerance): {epsilon:.8f}\n")

    root, history = newton_raphson_method(f, df, x0, epsilon, max_iter)

    if root is not None:
        print(f"\n✅ Root found: {root:.8f}")
        print(f"f(root) = {f(root):.8f}")

        # --- Plotting ---
        x_vals = np.linspace(1.5, 2.5, 400)
        y_vals = f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=r'$f(x) = x^3 - 2x - 5$', color='blue')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(root, color='red', linestyle='--', label=f'Root ≈ {root:.6f}')
        plt.scatter([root], [f(root)], color='red', zorder=5, s=60)

        # Plot tangent lines for first few iterations (for illustration)
        if history:
            for i, (n, xn, fxn, _) in enumerate(history[:4]):  # First 4 iterations
                dfxn = df(xn)
                if abs(dfxn) > 1e-8:
                    # Tangent line: y = f'(xn)(x - xn) + f(xn)
                    tangent_x = np.linspace(xn - 0.3, xn + 0.3, 2)
                    tangent_y = dfxn * (tangent_x - xn) + fxn
                    plt.plot(tangent_x, tangent_y, 'orange', linestyle='--', alpha=0.7)
                    plt.scatter([xn], [fxn], color='orange', s=30)

        plt.xlim(1.5, 2.5)
        plt.ylim(-2, 4)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Newton-Raphson Method: Solving $x^3 - 2x - 5 = 0$')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()