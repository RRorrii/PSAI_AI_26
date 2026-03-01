import numpy as np
import matplotlib.pyplot as plt

def load_custom_dataset():
    data = [
        [4, 1, 1],
        [-4, 1, 1],
        [4, -1, 1],
        [-4, -1, 0]
    ]
    X = np.array([[p[0], p[1]] for p in data], dtype=float)
    y = np.array([p[2] for p in data], dtype=int)
    y = np.where(y == 0, -1, 1)
    return X, y

def step_sign(x):
    return 1 if x >= 0 else -1

class ADALINE:
    def __init__(self, input_size, learning_rate=0.01):
        self.w = np.zeros(input_size, dtype=float)
        self.b = 0.0
        self.lr = learning_rate
        self.mse_history = []

    def predict_raw(self, x):
        return np.dot(self.w, x) + self.b

    def predict(self, x):
        return step_sign(self.predict_raw(x))

    def fit_adaline(self, X, y, epochs=50, shuffle=True):
        n = X.shape[0]
        for epoch in range(epochs):
            if shuffle:
                idx = np.random.permutation(n)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            mse = 0.0
            for xi, yi in zip(X_epoch, y_epoch):
                raw = self.predict_raw(xi)
                error = yi - raw
                self.w += self.lr * error * xi
                self.b += self.lr * error
                mse += error**2

            mse /= n
            self.mse_history.append(mse)
            print(f"Эпоха {epoch+1:2d}: MSE = {mse:.4f}")

    def decision_boundary(self, x_vals):
        if abs(self.w[1]) < 1e-12:
            return np.full_like(x_vals, np.nan)
        return (-(self.w[0] * x_vals + self.b) / self.w[1])

def learning_rate_study(X, y, rates, epochs=30):
    results = {}
    for lr in rates:
        model = ADALINE(input_size=2, learning_rate=lr)
        model.fit_adaline(X, y, epochs=epochs, shuffle=True)
        results[lr] = model.mse_history
    return results

def visualize(X, y, model, new_point=None):
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(range(1, len(model.mse_history)+1), model.mse_history, 'b-o')
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("Ошибка по эпохам")
    plt.grid(True)

    plt.subplot(1,2,2)
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:,0], X[:,1], c=colors, edgecolors='k', s=120)

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = model.decision_boundary(x_vals)
    if not np.isnan(y_vals).all():
        plt.plot(x_vals, y_vals, 'g--', linewidth=2, label='Разделяющая линия')

    if new_point is not None:
        p = np.array(new_point)
        pred = model.predict(p)
        color = 'blue' if pred == 1 else 'red'
        plt.scatter(p[0], p[1], c=color, s=200, marker='*', edgecolors='k')
        plt.text(p[0], p[1], f"Класс: {pred}", fontsize=10)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Классификация")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, y = load_custom_dataset()

    rates = [0.001, 0.01, 0.05, 0.1]
    lr_results = learning_rate_study(X, y, rates, epochs=40)

    model = ADALINE(input_size=2, learning_rate=0.05)
    model.fit_adaline(X, y, epochs=50)

    new_point = np.array([0.0, 0.0])
    visualize(X, y, model, new_point=new_point)

    preds = np.array([model.predict(x) for x in X])
    acc = np.mean(preds == y) * 100
    print(f"Веса: {model.w}, bias: {model.b:.4f}")
    print(f"Точность на обучающей выборке: {acc:.1f}%")
