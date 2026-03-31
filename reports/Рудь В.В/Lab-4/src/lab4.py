import numpy as np
import matplotlib.pyplot as plt
from itertools import product

n = 5
task = "OR"

def logic_function(x):
    if task == "OR":
        return int(any(x))
    raise ValueError("Unknown task")

X = np.array(list(product([0, 1], repeat=n)))
y = np.array([logic_function(x) for x in X]).reshape(-1, 1)

np.random.seed(1)
indices = np.random.permutation(len(X))
train_size = len(X) // 2

train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_perceptron(X, y, alpha=0.1, adaptive=False, max_epochs=5000):
    w = np.random.uniform(-1, 1, (n, 1))
    b = np.random.uniform(-1, 1)

    errors_train = []
    errors_test = []

    for epoch in range(max_epochs):
        net = X @ w + b
        out = sigmoid(net)

        error = y - out
        mse = np.mean(error ** 2)
        errors_train.append(mse)
        test_out = sigmoid(X_test @ w + b)
        test_mse = np.mean((y_test - test_out) ** 2)
        errors_test.append(test_mse)

        grad = error * sigmoid_derivative(out)

        if adaptive:
            alpha = 0.1 / (1 + 0.01 * epoch)

        w += alpha * (X.T @ grad)
        b += alpha * np.sum(grad)

        if mse < 1e-4:
            break

    return w, b, errors_train, errors_test, epoch + 1

w_fixed, b_fixed, err_train_fixed, err_test_fixed, epochs_fixed = train_perceptron(
    X_train, y_train, alpha=0.1, adaptive=False
)

w_adapt, b_adapt, err_train_adapt, err_test_adapt, epochs_adapt = train_perceptron(
    X_train, y_train, alpha=0.1, adaptive=True
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(err_train_fixed, label="Train (fixed α)")
plt.plot(err_test_fixed, label="Test (fixed α)")
plt.title("Фиксированный шаг обучения")
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(err_train_adapt, label="Train (adaptive α)")
plt.plot(err_test_adapt, label="Test (adaptive α)")
plt.title("Адаптивный шаг обучения")
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.legend()

plt.tight_layout()
plt.show()

def predict(x, w, b):
    prob = sigmoid(np.dot(x, w) + b)[0]
    return prob, int(prob >= 0.5)

print("\n=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
print(f"Эпох (фиксированный шаг): {epochs_fixed}")
print(f"Эпох (адаптивный шаг): {epochs_adapt}")

print("\nВесовые коэффициенты (фиксированный шаг):")
print(w_fixed.flatten())
print("Порог:", b_fixed)

print("\nВесовые коэффициенты (адаптивный шаг):")
print(w_adapt.flatten())
print("Порог:", b_adapt)

# Проверка на всех наборах
print("\n=== Проверка на полной таблице истинности ===")
for x in X:
    prob, cls = predict(x, w_adapt, b_adapt)
    print(f"{x} → P={prob:.3f}, class={cls}")
