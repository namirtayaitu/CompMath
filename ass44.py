import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# ВВЕДИ СВОИ ДАННЫЕ
# -----------------------------
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2.5, 6.9, 18.3, 50.1], dtype=float)

# Точка, где нужно найти значение
x_target = 2.5


# -----------------------------
# 1. ЛАГРАНЖ
# -----------------------------
def lagrange(x_data, y_data, x_val):
    n = len(x_data)
    result = 0.0
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x_val - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result


y_lagrange = lagrange(x, y, x_target)


# -----------------------------
# 2. НЬЮТОН ВПЕРЁД (∆)
# -----------------------------
def newton_forward(x_data, y_data, x_val):
    n = len(x_data)
    h = x_data[1] - x_data[0]
    if not np.allclose(np.diff(x_data), h):
        return None  # требует равномерных данных

    # Строим таблицу прямых разностей
    table = [y_data[:]]
    for k in range(1, n):
        prev = table[-1]
        next_row = [prev[i + 1] - prev[i] for i in range(len(prev) - 1)]
        table.append(next_row)

    u = (x_val - x_data[0]) / h
    result = table[0][0]
    term = 1.0
    for k in range(1, n):
        term *= (u - k + 1) / k
        result += term * table[k][0]
    return result


y_newton_fwd = newton_forward(x, y, x_target)


# -----------------------------
# 3. НЬЮТОН НАЗАД (∇)
# -----------------------------
def newton_backward(x_data, y_data, x_val):
    n = len(x_data)
    h = x_data[1] - x_data[0]
    if not np.allclose(np.diff(x_data), h):
        return None

    # Таблица обратных разностей
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y_data[i]
    for k in range(1, n):
        for i in range(k, n):
            table[i][k] = table[i][k - 1] - table[i - 1][k - 1]

    v = (x_val - x_data[-1]) / h
    result = table[n - 1][0]
    term = 1.0
    for k in range(1, n):
        term *= (v + k - 1) / k
        result += term * table[n - 1][k]
    return result


y_newton_bwd = newton_backward(x, y, x_target)


# -----------------------------
# 4. РАЗДЕЛЁННЫЕ РАЗНОСТИ (для неравномерных данных)
# -----------------------------
def newton_divided_diff(x_data, y_data, x_val):
    n = len(x_data)
    # Таблица разделённых разностей
    dd = np.zeros((n, n))
    dd[:, 0] = y_data
    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (x_data[i + j] - x_data[i])

    # Строим полином
    result = dd[0][0]
    prod = 1.0
    for i in range(1, n):
        prod *= (x_val - x_data[i - 1])
        result += dd[0][i] * prod
    return result


y_newton_dd = newton_divided_diff(x, y, x_target)


# -----------------------------
# 5. КУБИЧЕСКИЙ СПЛАЙН (natural: S''(x0) = S''(xn) = 0)
# -----------------------------
def cubic_spline_natural(x_data, y_data, x_val):
    n = len(x_data) - 1
    h = np.diff(x_data)
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (y_data[i + 1] - y_data[i]) - (3 / h[i - 1]) * (y_data[i] - y_data[i - 1])

    # Решаем трёхдиагональную систему для c_i (вторые производные)
    l = np.zeros(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)

    l[0] = 1
    mu[0] = 0
    z[0] = 0

    for i in range(1, n):
        l[i] = 2 * (x_data[i + 1] - x_data[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n] = 1
    z[n] = 0
    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)

    # Обратный ход
    for i in range(n - 1, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (y_data[i + 1] - y_data[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # Найти, в каком интервале находится x_val
    idx = np.searchsorted(x_data, x_val) - 1
    idx = np.clip(idx, 0, n - 1)

    dx = x_val - x_data[idx]
    return y_data[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


y_spline = cubic_spline_natural(x, y, x_target)

# -----------------------------
# ВЫВОД РЕЗУЛЬТАТОВ
# -----------------------------
print("=== Интерполяция в точке x =", x_target, "===")
print(f"Лагранж:               {y_lagrange:.6f}")
if y_newton_fwd is not None:
    print(f"Ньютон вперёд (∆):     {y_newton_fwd:.6f}")
else:
    print("Ньютон вперёд:         — (неравномерные x)")

if y_newton_bwd is not None:
    print(f"Ньютон назад (∇):      {y_newton_bwd:.6f}")
else:
    print("Ньютон назад:          — (неравномерные x)")

print(f"Разделённые разности:  {y_newton_dd:.6f}")
print(f"Кубический сплайн:     {y_spline:.6f}")

# -----------------------------
# ГРАФИК
# -----------------------------
x_fine = np.linspace(x.min(), x.max(), 200)
y_lagrange_fine = [lagrange(x, y, xi) for xi in x_fine]
y_spline_fine = [cubic_spline_natural(x, y, xi) for xi in x_fine]

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', s=80, label='Данные', zorder=5)
plt.plot(x_fine, y_lagrange_fine, '--', label='Лагранж')
plt.plot(x_fine, y_spline_fine, '-', label='Кубический сплайн')
plt.axvline(x_target, color='gray', linestyle=':', alpha=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Сравнение методов интерполяции')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()