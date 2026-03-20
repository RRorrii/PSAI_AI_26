import numpy as np
import matplotlib.pyplot as plt


class BinaryClassifier:
    def __init__(self, features_count):

        self.w = np.random.uniform(-0.1, 0.1, features_count + 1)

    def _add_bias(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def sigmoid(self, z):

        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        return self.sigmoid(Xb @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def train_mse(self, X, y, alpha=0.1, adaptive=False, Ee=0.01, max_epochs=1000):
        Xb = self._add_bias(X)
        y = np.array(y, dtype=float)
        history = []

        for epoch in range(max_epochs):
            for xi, target in zip(Xb, y):

                z = np.dot(xi, self.w)
                y_hat = self.sigmoid(z)


                error = y_hat - target


                gradient = error * (y_hat * (1 - y_hat)) * xi

                if adaptive:

                    current_alpha = 1.0 / (np.dot(xi, xi) + 1e-6)
                else:
                    current_alpha = alpha

                self.w -= current_alpha * gradient


            preds = self.sigmoid(Xb @ self.w)
            Es = 0.5 * np.mean((y - preds) ** 2)
            history.append(Es)

            if Es <= Ee:
                break
        return history

    def train_bce(self, X, y, alpha=0.1, adaptive=False, Ee=0.01, max_epochs=1000):
        Xb = self._add_bias(X)
        y = np.array(y, dtype=float)
        history = []

        for epoch in range(max_epochs):
            for xi, target in zip(Xb, y):
                y_hat = self.sigmoid(np.dot(xi, self.w))


                gradient = (y_hat - target) * xi

                if adaptive:
                    current_alpha = 1.0 / (np.dot(xi, xi) + 1e-6)
                else:
                    current_alpha = alpha

                self.w -= current_alpha * gradient


            y_hat_all = self.sigmoid(Xb @ self.w)
            eps = 1e-12
            Es = -np.mean(
                y * np.log(y_hat_all + eps) +
                (1 - y) * np.log(1 - y_hat_all + eps)
            )
            history.append(Es)

            if Es <= Ee:
                break
        return history




X_data = np.array([[3, 4], [-3, 4], [3, -4], [-3, -4]])
y_data = np.array([1, 1, 0, 0])


model_mse_f = BinaryClassifier(2);
h_mse_f = model_mse_f.train_mse(X_data, y_data, alpha=0.5)
model_mse_a = BinaryClassifier(2);
h_mse_a = model_mse_a.train_mse(X_data, y_data, adaptive=True)
model_bce_f = BinaryClassifier(2);
h_bce_f = model_bce_f.train_bce(X_data, y_data, alpha=0.1)
model_bce_a = BinaryClassifier(2);
h_bce_a = model_bce_a.train_bce(X_data, y_data, adaptive=True)



plt.figure(figsize=(10, 5))
plt.plot(h_mse_f, label="MSE (фикс)")
plt.plot(h_mse_a, label="MSE (адапт)")
plt.plot(h_bce_f, label="BCE (фикс)")
plt.plot(h_bce_a, label="BCE (адапт)")
plt.yscale('log')
plt.xlabel("Эпоха")
plt.ylabel("Ошибка (log scale)")
plt.title("Сходимость алгоритмов")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()


xmin, xmax = -6, 6
ymin, ymax = -6, 6
grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
grid_pred = model_bce_a.predict(grid_points).reshape(grid_x.shape)

plt.figure(figsize=(6, 6))
plt.contourf(grid_x, grid_y, grid_pred, alpha=0.2, colors=['red', 'blue'])
plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap='bwr_r', edgecolors='k')

# Линия w0 + w1*x + w2*y = 0  =>  y = -(w0 + w1*x) / w2
w = model_bce_a.w
lx = np.array([xmin, xmax])
ly = -(w[0] + w[1] * lx) / w[2]
plt.plot(lx, ly, 'k--', label="Разделяющая прямая")

plt.xlim(xmin, xmax);
plt.ylim(ymin, ymax)
plt.title("Результат классификации (BCE Адаптив)")
plt.legend()
plt.show()

print(
    f"Эпох до сходимости:\nMSE фикс: {len(h_mse_f)}\nMSE адапт: {len(h_mse_a)}\nBCE фикс: {len(h_bce_f)}\nBCE адапт: {len(h_bce_a)}")