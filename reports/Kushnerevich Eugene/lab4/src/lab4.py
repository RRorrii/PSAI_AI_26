import numpy as np
import matplotlib.pyplot as plt
from itertools import product

n = 6  

def logic_func(row):
    return int(not all(x == 1 for x in row))

all_inputs = np.array(list(product([0, 1], repeat = n)))
all_outputs = np.array([logic_func(row) for row in all_inputs]).reshape(-1, 1)

print("Таблица истинности:")
print(f"{'Входы':<20} {'Выход'}")
for row, out in zip(all_inputs, all_outputs):
    print(f"  {row.astype(int)}  ->  {int(out[0])}")

ones_idx  = np.where(all_outputs[:, 0] == 0)[0]
zeros_idx = np.where(all_outputs[:, 0] == 1)[0]

np.random.seed(42)
zeros_perm = np.random.permutation(zeros_idx)
split = int(0.75 * len(zeros_idx))

train_idx = np.concatenate([ones_idx, zeros_perm[:split]])
test_idx  = zeros_perm[split:]

X_train, y_train = all_inputs[train_idx], all_outputs[train_idx]
X_test,  y_test  = all_inputs[test_idx],  all_outputs[test_idx]

print(f"\nОбучающая выборка ({len(train_idx)} наборов): индексы {sorted(train_idx.tolist())}")
print(f"Тестовая выборка  ({len(test_idx)} наборов): индексы {sorted(test_idx.tolist())}")


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))

def linear_output(W, X, T):
    return X @ W - T

def bce_loss(y_pred, y_true):
    eps = 1e-12
    yp = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp))


def fixed_fit(X_tr, y_tr, X_te, y_te, alpha=0.1, max_epochs=10000, Ee=0.05):
    N, n = X_tr.shape
    W = np.zeros((n, 1))
    T = 0.0
    Es_train, Es_test = [], []
    stopped = max_epochs

    print(f"\n[Фиксированный] alpha={alpha}, Ee={Ee}")

    for ep in range(max_epochs):
        order = np.random.permutation(N)
        for idx in order:
            xi = X_tr[idx:idx+1]
            ti = float(y_tr[idx, 0])
            s  = float(linear_output(W, xi, T)[0, 0])
            yi = sigmoid(s)
            d  = yi - ti
            W -= alpha * xi.T * d
            T += alpha * d


        p_tr = sigmoid(linear_output(W, X_tr, T))
        es_tr = bce_loss(p_tr, y_tr)
        Es_train.append(es_tr)


        p_te = sigmoid(linear_output(W, X_te, T))
        es_te = bce_loss(p_te, y_te)
        Es_test.append(es_te)

        if ep % 500 == 0 and ep > 0:
            print(f"  эпоха {ep:5d} | train={es_tr:.5f} | test={es_te:.5f}")

        if es_tr <= Ee:
            stopped = ep + 1
            print(f" Остановка на эпохе {stopped}, train = {es_tr:.5f}\n")
            break
    else:
        print(f" Лимит {max_epochs} эпох\n")

    return W, T, Es_train, Es_test, stopped


def adaptive_fit(X_tr, y_tr, X_te, y_te, max_epochs=10000, Ee=0.05):
    N, n = X_tr.shape
    W = np.zeros((n, 1))
    T = 0.0
    Es_train, Es_test = [], []
    stopped = max_epochs

    print(f"[Адаптивный] α(t), Ee={Ee}")

    for ep in range(max_epochs):
        order = np.random.permutation(N)
        for idx in order:
            xi = X_tr[idx:idx+1]
            ti = float(y_tr[idx, 0])
            norm_sq = float(np.sum(xi ** 2))
            alpha_t = 1.0 / norm_sq if norm_sq > 1e-12 else 0.1

            s  = float(linear_output(W, xi, T)[0, 0])
            yi = sigmoid(s)
            d  = yi - ti
            W -= alpha_t * xi.T * d
            T += alpha_t * d

        p_tr = sigmoid(linear_output(W, X_tr, T))
        es_tr = bce_loss(p_tr, y_tr)
        Es_train.append(es_tr)

        p_te = sigmoid(linear_output(W, X_te, T))
        es_te = bce_loss(p_te, y_te)
        Es_test.append(es_te)

        if ep % 500 == 0 and ep > 0:
            print(f"  эпоха {ep:5d} | train={es_tr:.5f} | test={es_te:.5f}")

        if es_tr <= Ee:
            stopped = ep + 1
            print(f"  Остановка на эпохе {stopped}, train={es_tr:.5f}\n")
            break
    else:
        print(f"  Лимит {max_epochs} эпох\n")

    return W, T, Es_train, Es_test, stopped


Ee = 0.01
max_epochs = 10000

np.random.seed(42)
W_fix, T_fix, es_tr_fix, es_te_fix, ep_fix = fixed_fit(
    X_train, y_train, X_test, y_test, alpha=0.1, Ee=Ee, max_epochs = max_epochs)

np.random.seed(42)
W_adp, T_adp, es_tr_adp, es_te_adp, ep_adp = adaptive_fit(
    X_train, y_train, X_test, y_test, Ee=Ee, max_epochs = max_epochs)


print("Итоговые веса и порог:")
print(f" Фиксированный: W={W_fix.flatten()}, T={T_fix:.5f}, эпох={ep_fix}")
print(f" Адаптивный: W={W_adp.flatten()}, T={T_adp:.5f}, эпох={ep_adp}")


print("\nОценка на тестовой выборке (фиксированный):")
for xi, ti in zip(X_test, y_test):
    s = float(linear_output(W_fix, xi.reshape(1,-1), T_fix)[0, 0])
    prob = sigmoid(s)
    pred = 1 if prob >= 0.5 else 0
    ok   = "Yay" if pred == int(ti[0]) else "X"
    print(f"  {xi.astype(int)} -> ŷ = {prob:.6f} -> класс {pred} (эталон {int(ti[0])}) {ok}")

print("\nОценка на тестовой выборке (адаптивный):")
for xi, ti in zip(X_test, y_test):
    s = float(linear_output(W_adp, xi.reshape(1,-1), T_adp)[0, 0])
    prob = sigmoid(s)
    pred = 1 if prob >= 0.5 else 0
    ok   = "Yay" if pred == int(ti[0]) else "X"
    print(f"  {xi.astype(int)} → ŷ={prob:.6f} -> класс {pred} (эталон {int(ti[0])}) {ok}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, es_tr, es_te, ep, title, alpha_lbl in [
    (axes[0], es_tr_fix, es_te_fix, ep_fix, "Фиксированный шаг", "α=0.1"),
    (axes[1], es_tr_adp, es_te_adp, ep_adp, "Адаптивный шаг", "α(t)"),
]:
    epochs = np.arange(1, len(es_tr) + 1)
    ax.plot(epochs, es_tr, lw=2, color="#2563EB", label="Обучающая Es")
    ax.plot(epochs, es_te, lw=2, color="#DC2626", ls="--", label="Тестовая Es")
    ax.axhline(Ee, color="gray", lw=1.3, ls=":", label=f"Eₑ = {Ee}")
    ax.set_yscale("log")
    ax.set_xlabel("Эпоха", fontsize=11)
    ax.set_ylabel("BCE Loss", fontsize=11)
    ax.set_title(f"{title}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35)

plt.tight_layout()
plt.show()

print("\n" + "─" * 50)
print(f" Режим функционирования (модель: адаптивный)")
print(f" Введите {n} чисел (0 или 1) через пробел, или 'exit'\n")

while True:
    user_input = input(" -> ").strip()
    if user_input in ("exit", ""):
        break
    try:
        vals = list(map(int, user_input.split()))
        assert len(vals) == n and all(v in (0,1) for v in vals)

        point = np.array(vals, dtype=float).reshape(1, -1)
        s = float(linear_output(W_adp, point, T_adp)[0, 0])
        prob = sigmoid(s)
        cls = 1 if prob >= 0.5 else 0

        print(f"S = {s:+.5f}")
        print(f"ŷ = {prob:.5f}  (вероятность класса 1)")
        print(f"Класс = {cls}\n")

    except Exception:
        print(f" Ошибка. Введите ровно {n} чисел (0 или 1)\n")

print("Работа завершена.")
