import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def main():

    n = 5
    task = "OR"

    EE = 0.01
    MAX_EPOCHS = 5000

    ALPHA_MSE_FIXED = 0.1
    ALPHA_BCE_FIXED = 0.1

    ALPHA0_MSE_ADAPT = 0.1
    ALPHA0_BCE_ADAPT = 0.1
    K_ADAPT = 0.01


    def logic_function(x):
        if task == "OR":
            return int(any(x))
        raise ValueError("Unknown task")


    X = np.array(list(product([0, 1], repeat=n)))
    y = np.array([logic_function(x) for x in X]).reshape(-1, 1)

    print("=== ПОЛНАЯ ТАБЛИЦА ИСТИННОСТИ ===")
    for x_vec, y_val in zip(X, y):
        print(f"{x_vec} -> {int(y_val[0])}")


    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\n=== РАЗДЕЛЕНИЕ ВЫБОРКИ ===")
    print("Использован np.random.seed(42) для воспроизводимости.")
    print(f"Всего примеров: {len(X)}")
    print(f"Обучающая выборка (80%): {len(X_train)}")
    print(f"Тестовая выборка (20%): {len(X_test)}")


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def mse_loss_and_grad(y_true, y_pred):

        error = y_pred - y_true
        loss = 0.5 * error ** 2
        dnet = error * y_pred * (1 - y_pred)
        return float(loss), float(dnet)

    def bce_loss_and_grad(y_true, y_pred):

        eps = 1e-7
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -(y_true * np.log(y_pred_clipped) +
                 (1 - y_true) * np.log(1 - y_pred_clipped))
        dnet = y_pred - y_true
        return float(loss), float(dnet)


    def train_perceptron_online(
        X_train_, y_train_,
        X_test_, y_test_,
        loss_type="mse",
        alpha_init=0.1,
        adaptive=False,
        max_epochs=MAX_EPOCHS,
        ee=EE,
    ):
        m_train, n_features = X_train_.shape

        rng = np.random.default_rng(1)
        w = rng.uniform(-1, 1, size=(n_features, 1))
        b = float(rng.uniform(-1, 1))

        if loss_type == "mse":
            loss_grad_fn = mse_loss_and_grad
        elif loss_type == "bce":
            loss_grad_fn = bce_loss_and_grad
        else:
            raise ValueError("loss_type must be 'mse' or 'bce'")

        history_train = []
        history_test = []

        for epoch in range(max_epochs):
            if adaptive:
                alpha = alpha_init / (1 + K_ADAPT * epoch)
            else:
                alpha = alpha_init

            perm = np.random.permutation(m_train)
            X_train_shuffled = X_train_[perm]
            y_train_shuffled = y_train_[perm]

            epoch_loss_sum = 0.0


            for x_vec, y_true in zip(X_train_shuffled, y_train_shuffled):
                x_vec = x_vec.reshape(-1, 1)
                y_true = float(np.squeeze(y_true))

                net = (np.dot(x_vec.T, w) + b).item()
                y_pred = float(sigmoid(net))

                loss, dnet = loss_grad_fn(y_true, y_pred)
                epoch_loss_sum += loss

                w -= alpha * dnet * x_vec
                b -= alpha * dnet

            Es_train = epoch_loss_sum / m_train
            history_train.append(Es_train)


            test_loss_sum = 0.0
            for x_vec, y_true in zip(X_test_, y_test_):
                x_vec = x_vec.reshape(-1, 1)
                y_true = float(np.squeeze(y_true))

                net = (np.dot(x_vec.T, w) + b).item()
                y_pred = float(sigmoid(net))
                loss, _ = loss_grad_fn(y_true, y_pred)
                test_loss_sum += loss

            Es_test = test_loss_sum / len(X_test_)
            history_test.append(Es_test)

            if Es_train <= ee:
                break

        epochs_done = epoch + 1
        return w, b, np.array(history_train), np.array(history_test), epochs_done

    results = {}


    w_mse_fix, b_mse_fix, hist_mse_fix_tr, hist_mse_fix_te, ep_mse_fix = train_perceptron_online(
        X_train, y_train, X_test, y_test,
        loss_type="mse",
        alpha_init=ALPHA_MSE_FIXED,
        adaptive=False,
    )

    w_mse_ad, b_mse_ad, hist_mse_ad_tr, hist_mse_ad_te, ep_mse_ad = train_perceptron_online(
        X_train, y_train, X_test, y_test,
        loss_type="mse",
        alpha_init=ALPHA0_MSE_ADAPT,
        adaptive=True,
    )


    w_bce_fix, b_bce_fix, hist_bce_fix_tr, hist_bce_fix_te, ep_bce_fix = train_perceptron_online(
        X_train, y_train, X_test, y_test,
        loss_type="bce",
        alpha_init=ALPHA_BCE_FIXED,
        adaptive=False,
    )


    w_bce_ad, b_bce_ad, hist_bce_ad_tr, hist_bce_ad_te, ep_bce_ad = train_perceptron_online(
        X_train, y_train, X_test, y_test,
        loss_type="bce",
        alpha_init=ALPHA0_BCE_ADAPT,
        adaptive=True,
    )

    results["MSE_fixed"] = (w_mse_fix, b_mse_fix, hist_mse_fix_tr, hist_mse_fix_te, ep_mse_fix)
    results["MSE_adapt"] = (w_mse_ad, b_mse_ad, hist_mse_ad_tr, hist_mse_ad_te, ep_mse_ad)
    results["BCE_fixed"] = (w_bce_fix, b_bce_fix, hist_bce_fix_tr, hist_bce_fix_te, ep_bce_fix)
    results["BCE_adapt"] = (w_bce_ad, b_bce_ad, hist_bce_ad_tr, hist_bce_ad_te, ep_bce_ad)


    def predict_proba(X_, w_, b_):
        net = X_ @ w_ + b_
        return sigmoid(net)

    def accuracy(X_, y_, w_, b_):
        probs = predict_proba(X_, w_, b_)
        y_pred = (probs >= 0.5).astype(int)
        return np.mean(y_pred == y_)

    print("\n=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (n = 5, OR) ===")
    for name, (w_, b_, hist_tr, hist_te, epochs_done) in results.items():
        train_acc = accuracy(X_train, y_train, w_, b_)
        test_acc = accuracy(X_test, y_test, w_, b_)
        full_acc = accuracy(X, y, w_, b_)
        print(f"\n--- {name} ---")
        print(f"Эпох до останова: {epochs_done}")
        print(f"Accuracy train: {train_acc:.4f}")
        print(f"Accuracy test : {test_acc:.4f}")
        print(f"Accuracy full : {full_acc:.4f}")


    plt.figure(figsize=(10, 6))
    plt.plot(results["MSE_fixed"][2], label="MSE, фикс. шаг")
    plt.plot(results["MSE_adapt"][2], label="MSE, адаптивный шаг")
    plt.plot(results["BCE_fixed"][2], label="BCE, фикс. шаг")
    plt.plot(results["BCE_adapt"][2], label="BCE, адаптивный шаг")

    plt.xlabel("Эпоха")
    plt.ylabel("Средняя ошибка Es на обучении")
    plt.title("Сравнение MSE и BCE (онлайн‑обучение, n = 5, OR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    w_best, b_best, _, _, _ = results["BCE_adapt"]

    def predict_single(x_vec, w_, b_):
        x_vec = np.array(x_vec).reshape(1, -1)
        prob = float(predict_proba(x_vec, w_, b_)[0, 0])
        cls = int(prob >= 0.5)
        return prob, cls

    print("\n=== РЕЖИМ ФУНКЦИОНИРОВАНИЯ СЕТИ (BCE + адаптивный шаг) ===")
    print("Вводите 5 значений (0 или 1), разделённых пробелом. Для выхода введите 'q'.")

    while True:
        s = input("x1 x2 x3 x4 x5 = ").strip()
        if s.lower() == "q":
            print("Выход из режима функционирования.")
            break

        try:
            parts = s.split()
            if len(parts) != n:
                print(f"Нужно ввести ровно {n} значений.")
                continue

            x_input = [int(v) for v in parts]
            if any(v not in (0, 1) for v in x_input):
                print("Все значения должны быть 0 или 1.")
                continue

            prob, cls = predict_single(x_input, w_best, b_best)
            y_true = logic_function(x_input)

            print(f"Вход: {x_input}")
            print(f"Вероятность класса 1: {prob:.4f}")
            print(f"Предсказанный класс: {cls}")
            if cls == y_true:
                print("Совпадает с таблицей истинности.")
            else:
                print("Расхождение с таблицей истинности.")
        except Exception as e:
            print("Ошибка ввода, попробуйте ещё раз.", e)


if __name__ == "__main__":
    main()