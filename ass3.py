# Импортируем только то, что нужно
import numpy as np
import matplotlib.pyplot as plt

# Наши данные из задачи №6
x = [1, 2, 3, 4]
y = [2.5, 6.9, 18.3, 50.1]

# -----------------------------
# 1. Линейная модель: y = a*x + b
# -----------------------------
# Преобразуем в массивы для расчётов
x = np.array(x)
y = np.array(y)

# Считаем коэффициенты по формулам МНК (вручную!)
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x * x)

# Формулы:
# a = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
# b = (sum_y - a*sum_x) / n

a_lin = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
b_lin = (sum_y - a_lin * sum_x) / n

# Предсказания и ошибка
y_pred_lin = a_lin * x + b_lin
SSE_lin = np.sum((y - y_pred_lin) ** 2)

print("Линейная модель:")
print(f"  y = {a_lin:.2f} * x + {b_lin:.2f}")
print(f"  Ошибка (SSE) = {SSE_lin:.2f}")

# ----------------------------------------
# 2. Экспоненциальная модель: y = a * e^(b*x)
# ----------------------------------------
# Шаг 1: берём логарифм от y
Y = np.log(y)

# Шаг 2: считаем a' и b для уравнения Y = a' + b*x (тоже по формулам!)
sum_Y = np.sum(Y)
sum_xY = np.sum(x * Y)

b_exp = (n * sum_xY - sum_x * sum_Y) / (n * sum_x2 - sum_x**2)
a_prime = (sum_Y - b_exp * sum_x) / n  # это ln(a)

# Шаг 3: возвращаем a = e^(a')
a_exp = np.exp(a_prime)

# Предсказания и ошибка (в исходных y!)
y_pred_exp = a_exp * np.exp(b_exp * x)
SSE_exp = np.sum((y - y_pred_exp) ** 2)

print("\nЭкспоненциальная модель:")
print(f"  y = {a_exp:.2f} * exp({b_exp:.2f} * x)")
print(f"  Ошибка (SSE) = {SSE_exp:.4f}")

# -----------------
# Вывод
# -----------------
if SSE_exp < SSE_lin:
    print("\n✅ Экспоненциальная модель лучше!")
else:
    print("\n✅ Линейная модель лучше!")

# -----------------
# График
# -----------------
plt.scatter(x, y, color='red', label='Данные', s=100)
plt.plot(x, y_pred_lin, '--', label='Линейная')
plt.plot(x, y_pred_exp, '-', label='Экспоненциальная')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Сравнение моделей')
plt.show()