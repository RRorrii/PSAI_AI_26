import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [6, 6],
    [-6, 6],
    [6, -6],
    [-6, -6]
], dtype=float)

E = np.array([1 if x[0] > 0 else 0 for x in X], dtype=float)

etas = [0.001, 0.01, 0.05]
epochs = 80

all_mse_curves = {}

def train_perceptron(X, E, eta, epochs):
    w = np.random.randn(2)
    T = np.random.randn()

    mse_list = []

    for epoch in range(epochs):
        mse = 0
        for x, e in zip(X, E):

            S = np.dot(w, x) - T
            y_lin = S
            error = e - y_lin
            mse += error**2

            w += eta * error * x
            T -= eta * error

        mse_list.append(mse / len(X))

    return w, T, mse_list


for eta in etas:
    w, T, mse_curve = train_perceptron(X, E, eta, epochs)
    all_mse_curves[eta] = mse_curve

w_vis = w
T_vis = T

plt.figure(figsize=(7, 5))
for eta, curve in all_mse_curves.items():
    plt.plot(curve, label=f"eta = {eta}")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.title("График изменения ошибки MSE")
plt.grid()
plt.legend()
plt.show()
plt.figure(figsize=(7,7))

# обучающие точки
for i in range(len(X)):
    color = "red" if E[i] == 1 else "blue"
    plt.scatter(X[i,0], X[i,1], color=color, s=90)

# разделяющая линия
x_line = np.linspace(-7, 7, 300)
y_line = (T_vis - w_vis[0] * x_line) / w_vis[1]

plt.plot(x_line, y_line, 'k', linewidth=2)

plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.grid()
plt.title("Исходные точки и разделяющая линия")
plt.show()

def classify_and_plot(x1, x2):
    S = w_vis[0]*x1 + w_vis[1]*x2 - T_vis
    y = 1 if S > 0 else 0

    print("\n========== РЕЗУЛЬТАТ КЛАССИФИКАЦИИ ==========")
    print(f"Введённый вектор: ({x1}, {x2})")
    print(f"Взвешенная сумма S = {S:}")
    print(f"Класс сети: {y}")
    print("=============================================\n")

    plt.figure(figsize=(7,7))

    # обучающие точки
    for i in range(len(X)):
        color = "red" if E[i] == 1 else "blue"
        plt.scatter(X[i,0], X[i,1], color=color, s=90)

    # разделяющая линия
    plt.plot(x_line, y_line, 'k')

    # пользовательская точка
    plt.scatter(x1, x2, color="green", s=150, marker="x")

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.grid()
    plt.title("Классификация пользовательской точки")
    plt.show()

while True:
    print("\nВведите координаты точки, которую нужно классифицировать.")
    print("Чтобы выйти, введите 'exit'\n")

    x1 = input("x1 = ")
    if x1 == "exit":
        break
    x2 = input("x2 = ")
    if x2 == "exit":
        break

    try:
        x1 = float(x1)
        x2 = float(x2)
        classify_and_plot(x1, x2)
    except:
        print("Ошибка: введите число.")