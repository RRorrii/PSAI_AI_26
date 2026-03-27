import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def generate_truth_table_andi(n=8):
    X = np.array(list(product([0, 1], repeat=n)), dtype=float)
    y = np.ones((len(X), 1), dtype=float)
    y[np.all(X == 1, axis=1)] = 0.0
    return X, y


def split_train_test(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    zero_idx = np.where(y.ravel() == 0)[0]
    one_idx = np.where(y.ravel() == 1)[0]
    rng.shuffle(one_idx)
    test_size = max(1, int(len(one_idx) * test_ratio))
    test_idx = one_idx[:test_size]
    train_idx = np.concatenate([zero_idx, one_idx[test_size:]])
    rng.shuffle(train_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def sigmoid(s):
    s = np.clip(s, -50, 50)
    return 1.0 / (1.0 + np.exp(-s))


def bce_sum(y_pred, y_true, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


def accuracy_score(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


def alpha_adaptive(x):
    return 1.0 / (1.0 + float(np.sum(x ** 2)))


class SigmoidPerceptron:
    def __init__(self, n_inputs, seed=42, w_clip=50.0):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-0.5, 0.5, size=(n_inputs,))
        self.T = 0.0
        self.w_clip = float(w_clip)

    def s(self, x):
        return float(np.dot(self.w, x) - self.T)

    def y(self, x):
        return float(sigmoid(self.s(x)))

    def predict_proba(self, X):
        return np.array([self.y(x) for x in X], dtype=float).reshape(-1, 1)

    def predict_class(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def update_bce(self, x, e, alpha):
        y = self.y(x)
        err = y - e
        self.w = self.w - alpha * err * x
        self.T = self.T + alpha * err
        self.w = np.clip(self.w, -self.w_clip, self.w_clip)
        self.T = float(np.clip(self.T, -self.w_clip, self.w_clip))


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    mode="fixed",
    alpha_fixed=0.1,
    max_epochs=3000,
    target_error=0.05,
    shuffle=True,
    seed=42,
    verbose=False
):
    model = SigmoidPerceptron(n_inputs=X_train.shape[1], seed=seed, w_clip=50.0)
    rng = np.random.default_rng(seed)

    train_errors = []
    test_errors = []

    n = X_train.shape[0]

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)

        for i in idx:
            x = X_train[i]
            e = float(y_train[i, 0])

            if mode == "fixed":
                alpha = float(alpha_fixed)
            elif mode == "adaptive":
                alpha = alpha_adaptive(x)
            else:
                raise ValueError("mode must be 'fixed' or 'adaptive'")

            model.update_bce(x, e, alpha)

        y_train_prob = model.predict_proba(X_train)
        y_test_prob = model.predict_proba(X_test)

        train_error = bce_sum(y_train_prob, y_train)
        test_error = bce_sum(y_test_prob, y_test)

        train_errors.append(train_error)
        test_errors.append(test_error)

        if verbose and (epoch == 1 or epoch % 200 == 0 or epoch == max_epochs):
            print(
                f"Эпоха {epoch:4d} | "
                f"BCE_train = {train_error:.6f} | "
                f"BCE_test = {test_error:.6f}"
            )

        if train_error <= target_error:
            break

    history = {
        "train_errors": np.array(train_errors, dtype=float),
        "test_errors": np.array(test_errors, dtype=float),
        "epochs_used": len(train_errors),
    }

    return model, history


def run_fixed_alpha_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
    alpha_values,
    max_epochs=3000,
    target_error=0.05,
    seed=42
):
    results = []

    for alpha in alpha_values:
        print(f"\nЗапуск обучения для alpha = {alpha}")

        model, history = train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            mode="fixed",
            alpha_fixed=alpha,
            max_epochs=max_epochs,
            target_error=target_error,
            shuffle=True,
            seed=seed,
        )

        train_prob = model.predict_proba(X_train)
        test_prob = model.predict_proba(X_test)

        results.append({
            "alpha": alpha,
            "model": model,
            "history": history,
            "train_error": bce_sum(train_prob, y_train),
            "test_error": bce_sum(test_prob, y_test),
            "train_acc": accuracy_score(y_train, train_prob),
            "test_acc": accuracy_score(y_test, test_prob),
            "epochs_used": history["epochs_used"],
        })

    return results


def select_best_fixed_result(results):
    return sorted(results, key=lambda r: (r["test_error"], r["epochs_used"]))[0]


def print_experiment_table(results):
    print("\nТаблица результатов экспериментов")
    print("alpha | epochs | BCE_train | BCE_test | acc_train | acc_test")

    for r in results:
        print(
            f"{r['alpha']:.4f} | "
            f"{r['epochs_used']} | "
            f"{r['train_error']:.6f} | "
            f"{r['test_error']:.6f} | "
            f"{r['train_acc']:.4f} | "
            f"{r['test_acc']:.4f}"
        )


def print_model_parameters(model):
    print("\nПараметры модели")

    for i, wi in enumerate(model.w, start=1):
        print(f"w{i} = {wi:.6f}")

    print(f"Порог T = {model.T:.6f}")


def full_truth_table_accuracy(model, X, y):
    y_prob = model.predict_proba(X)
    y_pred = (y_prob >= 0.5).astype(int)
    return float(np.mean(y_pred == y))


def check_full_truth_table(model, X, y):
    y_prob = model.predict_proba(X)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(np.mean(y_pred == y))

    print(f"\nТочность на полной таблице: {acc:.6f}")

    wrong_idx = np.where(y_pred.ravel() != y.ravel())[0]
    if len(wrong_idx) == 0:
        print("Ошибок нет")
    else:
        print(f"Количество ошибок: {len(wrong_idx)}")
        for idx in wrong_idx[:10]:
            print(
                f"X = {X[idx].astype(int)}, "
                f"y_true = {int(y[idx, 0])}, "
                f"y_pred = {int(y_pred[idx, 0])}, "
                f"p = {y_prob[idx, 0]:.6f}"
            )


def plot_fixed_alpha_curves(fixed_results):
    plt.figure(figsize=(10, 6))

    for r in fixed_results:
        plt.plot(
            np.arange(1, len(r["history"]["train_errors"]) + 1),
            r["history"]["train_errors"],
            label=f"alpha={r['alpha']}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("BCE(train)")
    plt.title("Обучение при разных alpha")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_errors(history_fixed_best, history_adaptive):
    plt.figure(figsize=(10, 6))

    plt.plot(history_fixed_best["train_errors"], label="Fixed train")
    plt.plot(history_fixed_best["test_errors"], label="Fixed test")

    plt.plot(history_adaptive["train_errors"], label="Adaptive train")
    plt.plot(history_adaptive["test_errors"], label="Adaptive test")

    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.title("Ошибка обучения")
    plt.legend()
    plt.grid(True)
    plt.show()


def interactive_mode(model):
    print("\nРежим функционирования")
    print("Введите 8 чисел (0 или 1)")
    print("Для выхода нажмите q")

    while True:
        s = input("x1 x2 x3 x4 x5 x6 x7 x8 > ").strip()

        if s.lower() == "q":
            break

        parts = s.split()

        if len(parts) != 8:
            print("Нужно 8 чисел")
            continue

        try:
            x = np.array([int(v) for v in parts], dtype=float)
        except ValueError:
            print("Введите только 0 или 1")
            continue

        if not np.all(np.isin(x, [0, 1])):
            print("Введите только 0 или 1")
            continue

        p = model.y(x)
        cls = 1 if p >= 0.5 else 0

        print(f"Вероятность = {p:.6f}")
        print(f"Класс = {cls}")


def main():
    n = 8
    seed = 42
    test_ratio = 0.2
    target_error = 0.05
    max_epochs = 3000

    fixed_alpha_values = [0.01, 0.05, 0.10, 0.20, 0.50]

    X, y = generate_truth_table_andi(n)

    print("Всего наборов:", len(X))
    print("Единиц на выходе:", int(np.sum(y == 1)))
    print("Нулей на выходе:", int(np.sum(y == 0)))

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio=test_ratio, seed=seed)

    print("\nTrain:", len(X_train))
    print("Test:", len(X_test))

    print("\nЭксперимент A — фиксированный шаг")

    fixed_results = run_fixed_alpha_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        alpha_values=fixed_alpha_values,
        max_epochs=max_epochs,
        target_error=target_error,
        seed=seed
    )

    print_experiment_table(fixed_results)

    best_fixed = select_best_fixed_result(fixed_results)

    print("\nЛучшая fixed-модель:")
    print("alpha =", best_fixed["alpha"])
    print("epochs =", best_fixed["epochs_used"])

    print_model_parameters(best_fixed["model"])
    check_full_truth_table(best_fixed["model"], X, y)

    print("\nЭксперимент B — адаптивный шаг")

    model_adaptive, history_adaptive = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        mode="adaptive",
        max_epochs=max_epochs,
        target_error=target_error,
        seed=seed,
        verbose=True
    )

    print_model_parameters(model_adaptive)
    check_full_truth_table(model_adaptive, X, y)

    fixed_full_acc = full_truth_table_accuracy(best_fixed["model"], X, y)
    adaptive_full_acc = full_truth_table_accuracy(model_adaptive, X, y)

    fixed_test_error = best_fixed["test_error"]
    adaptive_test_error = bce_sum(model_adaptive.predict_proba(X_test), y_test)

    print("\nСравнение моделей")
    print(f"Fixed full accuracy = {fixed_full_acc:.6f}")
    print(f"Adaptive full accuracy = {adaptive_full_acc:.6f}")
    print(f"Fixed test BCE = {fixed_test_error:.6f}")
    print(f"Adaptive test BCE = {adaptive_test_error:.6f}")

    plot_fixed_alpha_curves(fixed_results)
    plot_errors(best_fixed["history"], history_adaptive)

    if adaptive_full_acc > fixed_full_acc:
        best_model = model_adaptive
        best_name = "адаптивный шаг"
    elif adaptive_full_acc < fixed_full_acc:
        best_model = best_fixed["model"]
        best_name = "фиксированный шаг"
    else:
        if adaptive_test_error < fixed_test_error:
            best_model = model_adaptive
            best_name = "адаптивный шаг"
        else:
            best_model = best_fixed["model"]
            best_name = "фиксированный шаг"

    print(f"\nДля режима функционирования выбрана модель: {best_name}")

    interactive_mode(best_model)


if __name__ == "__main__":
    main()