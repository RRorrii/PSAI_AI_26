import numpy as np
import matplotlib.pyplot as plt
import itertools

NUM_FEATURES = 6
TARGET_GATE = "OR"
SIG_SCALE = 1.0
LR_CONST = 0.1
MAX_ITERS = 5000
LOSS_LIMIT = 0.01
DECISION_BORDER = 0.5
TRAIN_FRACTION = 0.75


def act_sigmoid(z, c=SIG_SCALE):
    return 1.0 / (1.0 + np.exp(-c * z))


def build_truth_data(n_bits: int, gate: str):
    combos = list(itertools.product([0, 1], repeat=n_bits))
    X = np.array(combos, dtype=float)

    if gate == "AND":
        y = np.array([int(all(v)) for v in combos], dtype=float)
    elif gate == "OR":
        y = np.array([int(any(v)) for v in combos], dtype=float)
    elif gate == "XOR":
        y = np.array([int(sum(v) % 2 == 1) for v in combos], dtype=float)
    elif gate == "NAND":
        y = np.array([int(not all(v)) for v in combos], dtype=float)
    elif gate == "NOR":
        y = np.array([int(not any(v)) for v in combos], dtype=float)
    else:
        raise ValueError("Unknown logic gate")

    return X, y


def split_dataset(X, y, ratio=TRAIN_FRACTION, seed=42):
    np.random.seed(seed)
    idx_train, idx_test = [], []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        n_train = max(1, int(len(cls_idx) * ratio))
        idx_train.extend(cls_idx[:n_train])
        idx_test.extend(cls_idx[n_train:])

    return (
        X[np.array(idx_train)],
        y[np.array(idx_train)],
        X[np.array(idx_test)],
        y[np.array(idx_test)],
    )


def total_loss(X, y, w_vec, theta):
    err = 0.0
    for x_i, t_i in zip(X, y):
        z = w_vec @ x_i - theta
        y_hat = act_sigmoid(z)
        err += 0.5 * (y_hat - t_i) ** 2
    return err


def forward(X, w_vec, theta):
    z = X @ w_vec - theta
    probs = act_sigmoid(z)
    preds = (probs >= DECISION_BORDER).astype(int)
    return probs, preds


def score(true_y, pred_y):
    return np.mean(true_y == pred_y) * 100.0


def fit_perceptron(
    Xtr, Ytr, Xte, Yte,
    adaptive=False,
    lr=LR_CONST,
    max_iter=MAX_ITERS,
    goal=LOSS_LIMIT,
    seed=0
):
    np.random.seed(seed)
    dim = Xtr.shape[1]

    w_vec = np.random.uniform(0, 0.5, size=dim)
    theta = np.random.uniform(0, 0.5)

    print("\n--- Инициализация параметров ---")
    print(f"Начальные веса: {np.round(w_vec, 4)}")
    print(f"Начальный порог: {theta:.4f}\n")

    train_hist, test_hist = [], []

    for epoch in range(1, max_iter + 1):
        lr_t = 1.0 / epoch if adaptive else lr

        for x_i, t_i in zip(Xtr, Ytr):
            z = np.sum(w_vec * x_i) - theta
            y_hat = act_sigmoid(z)
            delta = y_hat - t_i

            w_vec -= lr_t * delta * x_i
            theta += lr_t * delta

        Ltr = total_loss(Xtr, Ytr, w_vec, theta)
        Lte = total_loss(Xte, Yte, w_vec, theta)

        train_hist.append(Ltr)
        test_hist.append(Lte)

        if epoch % 200 == 0 or epoch == 1:
            print(f"[Эпоха {epoch:4d}]  TrainLoss={Ltr:.6f}  TestLoss={Lte:.6f}  lr={lr_t:.5f}")

        if Ltr < goal:
            print(f"\n>>> Сходимость достигнута на эпохе {epoch}, ошибка={Ltr:.6f}")
            break
    else:
        print(f"\n>>> Достигнут предел итераций ({max_iter}), ошибка={train_hist[-1]:.6f}")

    print("\n--- Итоговые параметры ---")
    print(f"Финальные веса: {np.round(w_vec, 5)}")
    print(f"Финальный порог: {theta:.5f}")

    return w_vec, theta, train_hist, test_hist, epoch


def visualize(train_err, test_err, epochs, title, ax):
    ep_range = np.arange(1, len(train_err) + 1)
    ax.plot(ep_range, train_err, lw=2, color="blue", label="Train")
    ax.plot(ep_range, test_err, lw=2, color="red", linestyle="--", label="Test")
    ax.axhline(LOSS_LIMIT, color="gray", linestyle=":", label="Goal")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.4)
    ax.legend()


def run_interactive(w_vec, theta):
    print(f"\nВведите {NUM_FEATURES} бит (пример: 1 0 1 1 0 1). 'q' — выход.\n")

    while True:
        raw = input("Ввод > ").strip()
        if raw.lower() == "q":
            print("Завершение.")
            break

        try:
            bits = list(map(int, raw.split()))
            if len(bits) != NUM_FEATURES:
                print("Неверное количество значений.")
                continue
            if any(b not in (0, 1) for b in bits):
                print("Допустимы только 0 и 1.")
                continue

            x = np.array(bits, dtype=float)
            z = w_vec @ x - theta
            p = act_sigmoid(z)
            cls = int(p >= DECISION_BORDER)

            print(f"S = {z:.4f}")
            print(f"P(y=1) = {p:.6f}")
            print(f"Класс = {cls}\n")

        except ValueError:
            print("Ошибка ввода.\n")


def main():
    print("=" * 60)
    print(f"  Персептрон: {TARGET_GATE} ({NUM_FEATURES} входов)")
    print("=" * 60)

    X_all, y_all = build_truth_data(NUM_FEATURES, TARGET_GATE)

    print("\n--- Генерация таблицы истинности ---")
    print(f"Всего комбинаций: {len(y_all)}")
    print(f"Класс 1: {int(y_all.sum())}")
    print(f"Класс 0: {len(y_all) - int(y_all.sum())}")

    Xtr, Ytr, Xte, Yte = split_dataset(X_all, y_all)

    print("\n--- Разбиение выборки ---")
    print(f"Train: {len(Ytr)} образцов")
    print(f"Test:  {len(Yte)} образцов")

    print("\n--- Обучение: фиксированный шаг ---")
    w1, t1, tr1, te1, ep1 = fit_perceptron(Xtr, Ytr, Xte, Yte, adaptive=False)

    print("\n--- Обучение: адаптивный шаг ---")
    w2, t2, tr2, te2, ep2 = fit_perceptron(Xtr, Ytr, Xte, Yte, adaptive=True)

    p_tr1, c_tr1 = forward(Xtr, w1, t1)
    p_te1, c_te1 = forward(Xte, w1, t1)
    acc_tr = score(Ytr, c_tr1)
    acc_te = score(Yte, c_te1)

    p_tr2, c_tr2 = forward(Xtr, w2, t2)
    p_te2, c_te2 = forward(Xte, w2, t2)
    acc_tr2 = score(Ytr, c_tr2)
    acc_te2 = score(Yte, c_te2)

    print("\n--- Итоговая точность ---")
    print(f"Train accuracy (fixed):      {acc_tr:.2f}%")
    print(f"Test accuracy  (fixed):      {acc_te:.2f}%")
    print(f"Train accuracy (adaptive):   {acc_tr2:.2f}%")
    print(f"Test accuracy  (adaptive):   {acc_te2:.2f}%")

    print("\nПримеры предсказаний (первые 5):")
    for i in range(min(5, len(Xte))):
        print(f"{Xte[i].astype(int)} → y={int(Yte[i])}, pred={c_te1[i]}, p={p_te1[i]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    visualize(tr1, te1, ep1, "Фиксированный шаг", axes[0])
    visualize(tr2, te2, ep2, "Адаптивный шаг", axes[1])
    plt.tight_layout()
    plt.show()

    run_interactive(w1, t1)


if __name__ == "__main__":
    main()
