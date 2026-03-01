import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [ 6,  1],
    [-6,  1],
    [ 6, -1],
    [-6, -1]
], dtype=float)

y = np.array([0, 0, 0, 1], dtype=float)

def activation(z):
    return np.where(z >= 0, 1, 0)
def train_perceptron(X, y, lr=0.1, epochs=50):
    n_samples, n_features = X.shape
    W = np.random.randn(n_features)
    b = np.random.randn()

    mse_history = []

    for epoch in range(epochs):
        y_pred_linear = X @ W + b
        error = y_pred_linear - y
        mse = np.mean(error**2)
        mse_history.append(mse)
        dW = (2/n_samples) * X.T @ error
        db = (2/n_samples) * np.sum(error)
        W -= lr * dW
        b -= lr * db

    return W, b, mse_history

learning_rates = [0.01, 0.05, 0.1, 0.2]
histories = {}
weights_by_lr = {}

epochs = 50

for lr in learning_rates:
    W, b, mse_hist = train_perceptron(X, y, lr=lr, epochs=epochs)
    histories[lr] = mse_hist
    weights_by_lr[lr] = (W, b)

plt.figure(figsize=(8, 5))
for lr, hist in histories.items():
    plt.plot(hist, label=f"lr={lr}")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.title("Зависимость MSE от номера эпохи")
plt.grid(True)
plt.legend()
plt.show()

def plot_decision_boundary(W, b, title="Разделяющая линия"):
    plt.figure(figsize=(6, 6))

    for i in range(len(X)):
        color = 'red' if y[i] == 1 else 'blue'
        plt.scatter(X[i, 0], X[i, 1], color=color, s=80)

    x_vals = np.linspace(-7, 7, 200)
    if abs(W[1]) < 1e-6:
        x0 = -b / W[0]
        plt.axvline(x0, color='k', linestyle='--', label='Граница')
    else:
        y_vals = -(W[0] * x_vals + b) / W[1]
        plt.plot(x_vals, y_vals, 'k--', label='Граница')

    plt.xlim(-7, 7)
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()

W, b = weights_by_lr[0.1]
plot_decision_boundary(W, b, title="Разделяющая линия (lr=0.1)")

def classify_point(x1, x2, W, b):
    z = W[0] * x1 + W[1] * x2 + b
    return activation(z)

def plot_user_point(x1, x2, W, b):
    plt.figure(figsize=(6, 6))

    for i in range(len(X)):
        color = 'red' if y[i] == 1 else 'blue'
        plt.scatter(X[i, 0], X[i, 1], color=color, s=80, alpha=0.7)

    x_vals = np.linspace(-7, 7, 200)
    if abs(W[1]) < 1e-6:
        x0 = -b / W[0]
        plt.axvline(x0, color='k', linestyle='--', label='Граница')
    else:
        y_vals = -(W[0] * x_vals + b) / W[1]
        plt.plot(x_vals, y_vals, 'k--', label='Граница')

    cls = classify_point(x1, x2, W, b)
    color = 'green' if cls == 1 else 'purple'
    plt.scatter([x1], [x2], color=color, s=150, edgecolor='black',
                label=f"Точка пользователя → класс {int(cls)}")

    plt.xlim(-7, 7)
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.legend()
    plt.title("Классификация пользовательской точки")
    plt.show()

x1_user = float(input("Введите x1: "))
x2_user = float(input("Введите x2: "))

plot_user_point(x1_user, x2_user, W, b)
