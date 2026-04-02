import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, n_inputs):
        self.w = np.random.uniform(-0.1, 0.1, n_inputs + 1)

    def _add_bias(self, X):
        X = np.atleast_2d(X)
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -15, 15)))

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        return self.sigmoid(Xb @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def train(self, X_train, y_train, X_test, y_test, alpha=0.1, adaptive=False, max_epochs=1000, Ee=1e-4):
        Xb_train = self._add_bias(X_train)
        Xb_test = self._add_bias(X_test)

        train_history = []
        test_history = []

        for epoch in range(max_epochs):
            for xi, target in zip(Xb_train, y_train):
                y_hat = self.sigmoid(np.dot(xi, self.w))
                gradient = (y_hat - target) * xi  # Для BCE лосса

                curr_alpha = 1.0 / (np.dot(xi, xi) + 1e-6) if adaptive else alpha
                self.w -= curr_alpha * gradient

            # Считаем ошибки (MSE) для графиков
            train_preds = self.sigmoid(Xb_train @ self.w)
            test_preds = self.sigmoid(Xb_test @ self.w)

            err_train = 0.5 * np.mean((y_train - train_preds) ** 2)
            err_test = 0.5 * np.mean((y_test - test_preds) ** 2)

            train_history.append(err_train)
            test_history.append(err_test)

            if err_train <= Ee:
                break

        return train_history, test_history


n = 5
X_all = np.array([[int(b) for b in format(i, f'0{n}b')] for i in range(2 ** n)])

y_all = (np.sum(X_all, axis=1) == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)


model_f = Perceptron(n)
h_train_f, h_test_f = model_f.train(X_train, y_train, X_test, y_test, alpha=0.3)

model_a = Perceptron(n)
h_train_a, h_test_a = model_a.train(X_train, y_train, X_test, y_test, adaptive=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(h_train_f, label='Train (Fix)')
plt.plot(h_test_f, '--', label='Test (Fix)')
plt.title('Фиксированный шаг')
plt.xlabel('Эпоха');
plt.ylabel('MSE');
plt.legend();
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(h_train_a, label='Train (Adapt)')
plt.plot(h_test_a, '--', label='Test (Adapt)')
plt.title('Адаптивный шаг')
plt.xlabel('Эпоха');
plt.grid();
plt.legend()

plt.tight_layout()
plt.show()


print(f"Веса (w1-w5): {model_a.w[1:]}")
print(f"Порог (w0): {model_a.w[0]}")

test_vector = np.array([0, 0, 0, 0, 0])
prob = model_a.predict_proba(test_vector)[0]
print(f"\nТест на векторе {test_vector}:")
print(f"Вероятность: {prob:.4f}, Класс: {1 if prob > 0.5 else 0}")