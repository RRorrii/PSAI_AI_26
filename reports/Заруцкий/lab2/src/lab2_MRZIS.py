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

class ADALINESequential:
    def __init__(self, input_size, lr=0.01):
        self.w = np.zeros(input_size, dtype=float)
        self.b = 0.0
        self.lr = lr

    def predict_raw(self, x):
        return np.dot(self.w, x) + self.b

    def predict(self, x):
        return step_sign(self.predict_raw(x))

    def train_fixed(self, X, y, epochs=100, E_eps=1e-6, shuffle=True):
        n = X.shape[0]
        E_history = []
        for epoch in range(epochs):
            if shuffle:
                idx = np.random.permutation(n)
            else:
                idx = np.arange(n)
            E = 0.0
            for i in idx:
                xi = X[i]
                yi = y[i]
                raw = self.predict_raw(xi)
                err = yi - raw
                self.w += self.lr * err * xi
                self.b += self.lr * err
                E += 0.5 * err**2
            E_history.append(E)
            print(f"[Fixed] Эпоха {epoch+1:3d}: E = {E:.6f}")
            if E <= E_eps:
                break
        return E_history

    def train_adaptive_236(self, X, y, epochs=100, E_eps=1e-6, shuffle=True):
        n = X.shape[0]
        E_history = []
        for epoch in range(epochs):
            if shuffle:
                idx = np.random.permutation(n)
            else:
                idx = np.arange(n)
            E = 0.0
            for i in idx:
                xi = X[i]
                yi = y[i]
                raw = self.predict_raw(xi)
                err = yi - raw
                denom = 1.0 + np.dot(xi, xi)
                alpha_t = 1.0 / denom
                self.w += alpha_t * err * xi
                self.b += alpha_t * err
                E += 0.5 * err**2
            E_history.append(E)
            print(f"[Adaptive] Эпоха {epoch+1:3d}: E = {E:.6f}")
            if E <= E_eps:
                break
        return E_history

    def decision_boundary(self, x_vals):
        if abs(self.w[1]) < 1e-12:
            return np.full_like(x_vals, np.nan)
        return (-(self.w[0] * x_vals + self.b) / self.w[1])

def run_experiments():
    X, y = load_custom_dataset()
    max_epochs = 100
    E_eps = 1e-6

    model_fixed = ADALINESequential(input_size=2, lr=0.05)
    E_fixed = model_fixed.train_fixed(X, y, epochs=max_epochs, E_eps=E_eps, shuffle=True)

    model_adapt = ADALINESequential(input_size=2, lr=None)
    E_adapt = model_adapt.train_adaptive_236(X, y, epochs=max_epochs, E_eps=E_eps, shuffle=True)

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    ax = axes[0]
    ax.plot(range(1, len(E_fixed)+1), E_fixed, 'r-o', label='Фиксированный шаг (0.05)')
    ax.plot(range(1, len(E_adapt)+1), E_adapt, 'b-s', label='Адаптивный')
    ax.set_xlabel("Эпоха p")
    ax.set_ylabel("Суммарная ошибка E(p)")
    ax.set_title("Сходимость: фиксированный vs адаптивный")
    ax.legend()
    ax.grid(True)

    ax2 = axes[1]
    colors = ['red' if label == -1 else 'blue' for label in y]
    ax2.scatter(X[:,0], X[:,1], c=colors, edgecolors='k', s=120, label='Обучающие точки')

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    x_vals = np.linspace(x_min, x_max, 200)

    y_fixed = model_fixed.decision_boundary(x_vals)
    if not np.isnan(y_fixed).all():
        ax2.plot(x_vals, y_fixed, 'r--', linewidth=2, label='Разделяющая (фикс.)')

    y_adapt = model_adapt.decision_boundary(x_vals)
    if not np.isnan(y_adapt).all():
        ax2.plot(x_vals, y_adapt, 'b-.', linewidth=2, label='Разделяющая (адапт)')

    for xi, yi in zip(X, y):
        pred_f = model_fixed.predict(xi)
        pred_a = model_adapt.predict(xi)
        

    new_point = np.array([0.0, 0.0])
    pred_f_new = model_fixed.predict(new_point)
    pred_a_new = model_adapt.predict(new_point)
    ax2.scatter(new_point[0], new_point[1], c='green', s=200, marker='*', edgecolors='k',
                label=f'Новая точка')

    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_title("Точки и разделяющие прямые (оба варианта)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    preds_fixed = np.array([model_fixed.predict(x) for x in X])
    acc_fixed = np.mean(preds_fixed == y) * 100
    preds_adapt = np.array([model_adapt.predict(x) for x in X])
    acc_adapt = np.mean(preds_adapt == y) * 100

    print("Фиксированный шаг: веса =", model_fixed.w, "bias =", model_fixed.b, f"Точность = {acc_fixed:.1f}%")
    print("Адаптивный: веса =", model_adapt.w, "bias =", model_adapt.b, f"Точность = {acc_adapt:.1f}%")

if __name__ == "__main__":
    run_experiments()