import numpy as np
import matplotlib.pyplot as plt


x = np.array([
    [ 5,  6],
    [-5,  6],
    [ 5, -6],
    [-5, -6]
])

etalon_value = np.array([0, 1, 1, 1]).reshape(-1, 1)


def linear_output(weights, x, T):
    return x @ weights - T


def online_fit(X, y, alpha=0.001, epochs=2500, eps=1e-7, mix=True):
    N = X.shape[0]
    W = np.zeros((X.shape[1], 1))
    T = 0
    mse = []
    prev_loss = np.inf

    print(f"Старт обучения (alpha = {alpha})")

    for ep in range(epochs):
        order = np.random.permutation(N) if mix else np.arange(N)

        for idx in order:
            x_i = X[idx:idx+1]
            t_i = y[idx]

            s = linear_output(W, x_i, T)
            delta = s - t_i

            W -= alpha * x_i.T @ delta
            T += alpha * delta.item()

        preds = linear_output(W, X, T)
        loss = np.mean((preds - y) ** 2)
        mse.append(loss)

        if ep % 100 == 0 and ep > 0:
            print(f"  эпоха {ep:4d} | MSE = {loss:.8f}")

        if ep > 30 and abs(loss - prev_loss) < eps:
            print(f"  остановка на эпохе {ep}")
            break

        prev_loss = loss

    print(f" Обучение завершено | итоговая MSE = {loss:.8f}\n")
    return W, T, mse


learning_rates = [0.0003, 0.0007, 0.0012, 0.002]
models = {}

for alpha in learning_rates:
    W, T, err = online_fit(x, etalon_value, alpha=alpha)
    models[alpha] = (W, T, err)


plt.figure(figsize=(10, 5))
for lr, (_, _, err) in models.items():
    plt.plot(err, label=f"lr={lr}", lw=1.5)

plt.yscale("log")
plt.xlabel("Эпохи")
plt.ylabel("MSE")
plt.grid(alpha=0.35)
plt.legend()
plt.tight_layout()
plt.show()


best_alpha = min(models, key=lambda k: models[k][2][-1])
W, T = models[best_alpha][0], models[best_alpha][1]
print(f" Выбрана модель с alpha = {best_alpha}\n")


added_pts = []
added_cls = []

print("Введите координаты точки (x_1 x_2), либо exit\n")

while True:
    user_input = input(" -> ").strip()
    if user_input in ("exit", ""):
        break

    try:
        x1_val, x2_val = map(float, user_input.split())
        point = np.array([[x1_val, x2_val]])

        score = linear_output(W, point, T)[0, 0]
        cls = 1 if score >= 0.5 else 0
        color_name = "красный" if cls == 1 else "синий"

        print(f"  Значение S = {score:>9.6f}")
        print(f"  Класс     = {cls}")

        added_pts.append([x1_val, x2_val])
        added_cls.append(cls)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(
            x[etalon_value[:,0]==0][:,0],
            x[etalon_value[:,0]==0][:,1],
            s=150, color="royalblue", edgecolors="black", label="класс 0"
        )

        ax.scatter(
            x[etalon_value[:,0]==1][:,0],
            x[etalon_value[:,0]==1][:,1],
            s=150, color="crimson", edgecolors="black", label="класс 1"
        )

        w1, w2 = W.flatten()
        xs = np.linspace(-9, 9, 400)
        if abs(w2) > 1e-8:
            ys = (T + 0.5 - w1 * xs) / w2
            ax.plot(xs, ys, "green", lw=2.3, label="граница")

        for p, c in zip(added_pts, added_cls):
            ax.scatter(p[0], p[1],
                       marker="X",
                       s=220,
                       color="crimson" if c else "royalblue",
                       edgecolors="black")

        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    except Exception:
        print("Ошибка ввода. Используйте формат: x y\n")

print("Работа программы завершена.")
