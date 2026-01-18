import numpy as np
import matplotlib.pyplot as plt

def fixed_point_method(g, x0, epsilon, max_iterations):
    history = []
    x_prev = x0

    print(f"{'Iter':<5} {'x_n':<14} {'g(x_n)':<14} {'|x_{{n+1}}-x_n|':<16}")
    print("-" * 60)

    for n in range(max_iterations):
        x_next = g(x_prev)
        error = abs(x_next - x_prev)

        history.append((n, x_prev, x_next, error))
        print(f"{n:<5} {x_prev:<14.8f} {x_next:<14.8f} {error:<16.8f}")

        if error < epsilon:
            return x_next, history

        x_prev = x_next

    print(f"Failed to converge after {max_iterations} iterations")
    return None, history

def f(x):
    return x ** 3 - 2 * x - 5

def g(x):
    return (2 * x + 5) ** (1 / 3)

if __name__ == "__main__":
    x0 = 2.0
    epsilon = 0.000001
    max_iter = 50

    print("Solving x^3 - 2x - 5 = 0 using Fixed-Point Iteration")
    print(f"Rearranged as: x = (2x + 5)^(1/3)")
    print(f"Initial guess x0 = {x0}")
    print(f"Epsilon (tolerance): {epsilon:.8f}\n")

    root, history = fixed_point_method(g, x0, epsilon, max_iter)

    if root is not None:
        print(f"\n✅ Fixed point found: {root:.8f}")
        print(f"f(root) = {f(root):.8f}")

        # --- Plotting ---
        x_vals = np.linspace(1.5, 2.5, 400)
        y_f = f(x_vals)
        y_g = g(x_vals)
        y_x = x_vals

        plt.figure(figsize=(10, 6))

        # Plot f(x) for context
        plt.subplot(1, 2, 1)
        plt.plot(x_vals, y_f, label=r'$f(x) = x^3 - 2x - 5$', color='blue')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(root, color='red', linestyle='--', label=f'Root ≈ {root:.6f}')
        plt.scatter([root], [f(root)], color='red', zorder=5)
        plt.title('Original Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Plot g(x) and y=x for fixed-point visualization
        plt.subplot(1, 2, 2)
        plt.plot(x_vals, y_g, label=r'$g(x) = \sqrt[3]{2x + 5}$', color='green')
        plt.plot(x_vals, y_x, label=r'$y = x$', color='black', linestyle='--')
        plt.axvline(root, color='red', linestyle='--', label=f'Fixed point ≈ {root:.6f}')
        plt.scatter([root], [g(root)], color='red', zorder=5)

        # Show iteration steps (staircase plot)
        if history:
            x_path = []
            y_path = []
            current = x0
            for i in range(min(10, len(history))):  # limit to first 10 steps for clarity
                next_val = g(current)
                x_path.extend([current, current, next_val])
                y_path.extend([current, next_val, next_val])
                current = next_val
            plt.plot(x_path, y_path, 'm--', alpha=0.7, linewidth=1, label='Iteration path')

        plt.title('Fixed-Point Iteration: $x = g(x)$')
        plt.xlabel('x')
        plt.ylabel('g(x)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.show()