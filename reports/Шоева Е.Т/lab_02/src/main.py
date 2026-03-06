import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [6, 2],
    [-6, 2],
    [6, -2],
    [-6, -2]
], dtype=float)

y = np.array([0, 0, 1, 0], dtype=float)

X = X / 6.0

X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

Ee = 0.001
max_epochs = 1000
lr_fixed = 0.1

def net_output(X, w):
    return np.dot(X, w)

def mse(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)

w_fixed = np.random.randn(3) * 0.1
Es_fixed = []
epoch_fixed = 0

while True:
    y_pred = net_output(X_bias, w_fixed)
    error = y - y_pred
    Es = mse(y, y_pred)
    Es_fixed.append(Es)

    if Es <= Ee or epoch_fixed >= max_epochs:
        break

    w_fixed = w_fixed + lr_fixed * np.dot(X_bias.T, error)

    epoch_fixed += 1

print("Фиксированный шаг: число эпох =", epoch_fixed)

w_adapt = np.random.randn(3) * 0.1
Es_adapt = []
epoch_adapt = 0

while True:
    Es_sum = 0

    for i in range(len(X_bias)):
        x_i = X_bias[i]
        y_i = y[i]

        y_pred = np.dot(x_i, w_adapt)
        error = y_i - y_pred

        alpha = 1 / (1 + np.sum(x_i[:-1] ** 2))

        w_adapt = w_adapt + alpha * error * x_i

    y_all = net_output(X_bias, w_adapt)
    Es = mse(y, y_all)
    Es_adapt.append(Es)

    if Es <= Ee or epoch_adapt >= max_epochs:
        break

    epoch_adapt += 1

print("Адаптивный шаг: число эпох =", epoch_adapt)

plt.figure(figsize=(9, 6))
plt.plot(Es_fixed, label="Фиксированный шаг (lr=0.1)")
plt.plot(Es_adapt, label="Адаптивный шаг α(t)")
plt.xlabel("Номер эпохи p")
plt.ylabel("Es")
plt.title("Зависимость суммарной ошибки Es от номера эпохи")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 7))

X_plot = X * 6

for i in range(len(X_plot)):
    if y[i] == 0:
        plt.scatter(X_plot[i][0], X_plot[i][1], color='blue', label='Класс 0' if i == 0 else "")
    else:
        plt.scatter(X_plot[i][0], X_plot[i][1], color='red', label='Класс 1')

x_vals = np.linspace(-8, 8, 200)
x_vals_norm = x_vals / 6

y_vals = -(w_adapt[0] * x_vals_norm + w_adapt[2]) / w_adapt[1]
y_vals = y_vals * 6

plt.plot(x_vals, y_vals, color='green', label="Разделяющая линия (адаптивный метод)")

x1 = float(input("Введите x1: "))
x2 = float(input("Введите x2: "))

user_norm = np.array([x1 / 6, x2 / 6, 1])
net = np.dot(user_norm, w_adapt)
user_class = 1 if net >= 0.5 else 0

print("Класс точки:", user_class)

if user_class == 0:
    plt.scatter(x1, x2, color='cyan', s=120, marker='x', label="Ваша точка (класс 0)")
else:
    plt.scatter(x1, x2, color='magenta', s=120, marker='x', label="Ваша точка (класс 1)")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Классификация (адаптивный шаг обучения)")
plt.legend()
plt.grid(True)
plt.show()
