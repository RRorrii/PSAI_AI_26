import numpy as np
import matplotlib.pyplot as plt
import itertools

N = 7

X_all = np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)
y_all = np.all(X_all == 1, axis=1).astype(float)

print(f"Всего наборов: {len(X_all)}")
print(f"Класс 1 (AND=1): {int(y_all.sum())}  |  Класс 0 (AND=0): {int((1-y_all).sum())}")

np.random.seed(42)
idx_pos = np.where(y_all == 1)[0]
idx_neg = np.where(y_all == 0)[0]

neg_perm  = np.random.permutation(idx_neg)
split_neg = int(0.7 * len(idx_neg))
train_neg, test_neg = neg_perm[:split_neg], neg_perm[split_neg:]

train_idx = np.concatenate([idx_pos, train_neg])
test_idx  = test_neg
np.random.shuffle(train_idx)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

print(f"Обучающая: {len(X_train)} (класс 1: {int(y_train.sum())})")
print(f"Тестовая:  {len(X_test)}  (класс 1: {int(y_test.sum())})")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def total_error(X, y, w, b, w_pos=1.0):
    yp = sigmoid(X @ w + b)
    wts = np.where(y == 1, w_pos, 1.0)
    return 0.5 * np.sum(wts * (y - yp) ** 2)

def accuracy(X, y, w, b):
    return np.mean((sigmoid(X @ w + b) >= 0.5).astype(float) == y)

def train_perceptron(X_tr, y_tr, X_te, y_te,
                     alpha_mode='fixed', alpha=0.5,
                     max_epochs=6000, tol=1e-4,
                     w_pos=127.0):
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 1.0, size=N)
    b = -float(N) + 0.5

    train_errors, test_errors = [], []
    lr = alpha
    prev_err = float('inf')

    for epoch in range(max_epochs):
        for i in rng.permutation(len(X_tr)):
            xi, yi = X_tr[i], y_tr[i]
            out = sigmoid(xi @ w + b)
            weight_i = w_pos if yi == 1 else 1.0
            delta = weight_i * (yi - out) * out * (1 - out)
            w += lr * delta * xi
            b += lr * delta

        err_tr = total_error(X_tr, y_tr, w, b, w_pos)
        err_te = total_error(X_te, y_te, w, b, w_pos)
        train_errors.append(err_tr)
        test_errors.append(err_te)

        if alpha_mode == 'adaptive':
            lr = min(lr * 1.05, 2.0) if err_tr < prev_err else max(lr * 0.5, 1e-5)

        prev_err = err_tr
        if err_tr < tol:
            print(f"  Сошлось за {epoch+1} эпох (ошибка={err_tr:.6f})")
            break
    else:
        print(f"  Достигнут лимит {max_epochs} эпох (ошибка={train_errors[-1]:.6f})")

    return w, b, train_errors, test_errors

print("\n=== Эксперимент А: фиксированный шаг α=0.5 ===")
wA, bA, tr_errA, te_errA = train_perceptron(
    X_train, y_train, X_test, y_test,
    alpha_mode='fixed', alpha=0.5)
print(f"  Точность на обучающей: {accuracy(X_train, y_train, wA, bA)*100:.1f}%")
print(f"  Точность на тестовой:  {accuracy(X_test,  y_test,  wA, bA)*100:.1f}%")
print(f"  Точность на ПОЛНОЙ:    {accuracy(X_all,   y_all,   wA, bA)*100:.1f}%")


print("\n=== Эксперимент Б: адаптивный шаг (нач. α=0.3) ===")
wB, bB, tr_errB, te_errB = train_perceptron(
    X_train, y_train, X_test, y_test,
    alpha_mode='adaptive', alpha=0.3)
print(f"  Точность на обучающей: {accuracy(X_train, y_train, wB, bB)*100:.1f}%")
print(f"  Точность на тестовой:  {accuracy(X_test,  y_test,  wB, bB)*100:.1f}%")
print(f"  Точность на ПОЛНОЙ:    {accuracy(X_all,   y_all,   wB, bB)*100:.1f}%")
print("\n=== Итоговые веса (Эксперимент А) ===")
for i, wi in enumerate(wA):
    print(f"  w{i+1} = {wi:+.4f}")
print(f"  b  = {bA:+.4f}")

print("\n=== Итоговые веса (Эксперимент Б) ===")
for i, wi in enumerate(wB):
    print(f"  w{i+1} = {wi:+.4f}")
print(f"  b  = {bB:+.4f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Лабораторная работа №4  |  Вариант 10: AND(n=7)", fontsize=14, fontweight='bold')

for ax, tr_e, te_e, title in [
    (axes[0], tr_errA, te_errA, f"А. Фиксированный шаг (α=0.5)\nЭпох: {len(tr_errA)}"),
    (axes[1], tr_errB, te_errB, f"Б. Адаптивный шаг (нач. α=0.3)\nЭпох: {len(tr_errB)}"),
]:
    ax.plot(range(1, len(tr_e)+1), tr_e, label="Обучающая", color="royalblue", lw=1.5)
    ax.plot(range(1, len(te_e)+1), te_e, label="Тестовая",  color="tomato", ls="--", lw=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Суммарная ошибка (SSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("convergence.png", dpi=150)
plt.close()
print("\nГрафик сохранён: convergence.png")

def predict(x_vec, w, b):
    x = np.array(x_vec, dtype=float)
    prob = float(sigmoid(x @ w + b))
    return prob, int(prob >= 0.5)

print("\n" + "="*55)
print("       РЕЖИМ ФУНКЦИОНИРОВАНИЯ (используются веса А)")
print("="*55)
print(f"Введите {N} чисел через пробел (только 0 или 1).")
print("Для выхода введите 'q'.\n")

while True:
    raw = input(f"Вход [{N} бит]: ").strip()
    if raw.lower() == 'q':
        print("Выход из режима функционирования.")
        break

    parts = raw.split()
    if len(parts) != N:
        print(f"  ✗  Ошибка: нужно ровно {N} значений, получено {len(parts)}.\n")
        continue

    try:
        vec = [int(p) for p in parts]
    except ValueError:
        print("  ✗  Ошибка: вводите только целые числа 0 или 1.\n")
        continue

    if any(v not in (0, 1) for v in vec):
        print("  ✗  Ошибка: допустимы только значения 0 и 1.\n")
        continue

    prob, cls = predict(vec, wA, bA)
    true_val  = int(all(v == 1 for v in vec))
    ok = "✓" if cls == true_val else "✗"
    print(f"  {ok}  Вероятность (класс 1): {prob:.4f}")
    print(f"     Результат AND{vec} = {cls}  (истинное значение: {true_val})\n")