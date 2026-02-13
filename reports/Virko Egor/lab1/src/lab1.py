import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, input_size=0, learning_rate=0.1):
        self.X = np.array([])
        self.w = np.random.uniform(0, 1.0, input_size + 1)
        self.learning_rate = learning_rate
        self.target = np.array([])

    def set_X(self, X: np.array) -> None:
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            return

        self.X = X
        self.X = np.insert(self.X, 0, -1, axis=1)

    def set_target(self, target: np.array) -> None:
        if self.X.ndim != 2:
            return

        if len(target) != len(self.X):
            return

        self.target = target

    def get_wsum(self, X: np.array) -> np.array:
        return np.dot(X, self.w)

    def activate(self, arr_wsum: np.array) -> np.array:
        return np.where(arr_wsum > 0, 1, 0)

    def prediction(self, X_input=None) -> np.array:
        X = self.X if X_input is None else X_input

        if X.size == 0:
            return None

        wsum = self.get_wsum(X)
        y = self.activate(wsum)
        return y

    def delta(self, error: np.array) -> None:
        self.w = self.w - self.learning_rate * np.dot(error, self.X)

    def mse(self, error: np.array) -> np.array:
        return np.mean(0.5 * error ** 2)

    def train(self, epochs=500) -> np.array:
        mse_history = []

        for epoch in range(epochs):
            s = self.get_wsum(self.X)
            error = s - self.target
            mse = self.mse(error)
            mse_history.append(mse)
            self.delta(error)

        return mse_history


X_train = np.array([[2, 4],
                    [-2, 4],
                    [2, -4],
                    [-2, -4]])

e_targets = np.array([0, 0, 1, 1])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
for eta in [0.0001, 0.001, 0.01]:
    model = Perceptron(input_size=2, learning_rate=eta)
    model.set_X(X_train)
    model.set_target(e_targets)
    history = model.train(epochs=500)
    plt.plot(history, label=f'η = {eta}')

plt.title("Зависимость MSE от номера эпохи")
plt.xlabel("Номер эпохи")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)

p = Perceptron(input_size=2, learning_rate=0.001)
p.set_X(X_train)
p.set_target(e_targets)
p.train(epochs=1000)


def plot_current_state(user_point=None, user_class=None):
    plt.subplot(1, 2, 1)

    xx, yy = np.meshgrid(np.linspace(-7, 7, 100),
                         np.linspace(-7, 7, 100))

    grid_points = np.c_[np.ones(xx.ravel().shape) * -1,
    xx.ravel(),
    yy.ravel()]

    Z_linear = p.get_wsum(grid_points).reshape(xx.shape)
    Z_class = p.prediction(grid_points).reshape(xx.shape)

    plt.contourf(xx, yy, Z_class,
                 levels=[-0.1, 0.5, 1.1],
                 colors=["#ffebee", "#e3f2fd"],
                 alpha=0.7)

    plt.contour(xx, yy, Z_linear,
                levels=[0],
                colors='black',
                linewidths=2)

    plt.scatter(X_train[:, 0],
                X_train[:, 1],
                c=e_targets,
                cmap='bwr',
                edgecolors='k',
                s=120)

    if user_point is not None:
        plt.scatter(user_point[0],
                    user_point[1],
                    color='yellow',
                    marker='*',
                    s=250,
                    edgecolors='black')

    plt.xlim([-7, 7])
    plt.ylim([-7, 7])
    plt.grid(True)
    plt.title("Разделяющая прямая S = 0")
    plt.xlabel("X1")
    plt.ylabel("X2")


plot_current_state()
plt.show(block=False)

print("\nВведите X1 X2 через пробел (или exit для выхода)\n")

try:
    while True:
        line = input("X1 X2: ")

        if line.lower() in ['exit', 'выход']:
            break

        parts = line.split()
        if len(parts) != 2:
            continue

        coords = [float(parts[0]), float(parts[1])]
        user_x = np.array([[-1, coords[0], coords[1]]])

        pred = p.prediction(user_x)[0]

        print(f"Класс: {pred} (0 = верхняя область, 1 = нижняя область)")

        plt.subplot(1, 2, 1)
        plt.cla()
        plot_current_state(coords, pred)
        plt.draw()
        plt.pause(0.1)

except ValueError:
    pass

plt.show()
