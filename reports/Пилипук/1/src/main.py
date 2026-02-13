import numpy as np
import matplotlib.pyplot as plt

X = np.array([[6, 1], [-6, 1], [6, -1], [-6, -1]])
e = np.array([0, 0, 0, 1]).reshape(-1, 1)

def predict(weights, X, T):
    return X @ weights - T

def MSE(e, y):
    return np.mean((y - e) ** 2)

def train(X, e, alpha, epochs=10000, tol=1e-6):
    N = X.shape[0]
    weights = np.zeros((X.shape[1], 1))
    T = 0.0
    errors = []

    for epoch in range(epochs):
        y = predict(weights, X, T)
        error = MSE(e, y)
        errors.append(error)

        if len(errors) > 1 and abs(errors[-1] - errors[-2]) < tol:
            print(f"MSE minimized at epoch = {epoch}")
            break

        grad_w = (X.T @ (y - e)) / N
        grad_T = np.mean(y - e)

        weights -= alpha * grad_w
        T += alpha * grad_T

    return weights, T, errors

alphas = [0.01, 0.001, 0.0001]
errors = []
weights_list = []
T_list = []

for alpha in alphas:
    weights, T, error = train(X, e, alpha=alpha)
    weights_list.append(weights)
    T_list.append(T)
    errors.append(error)
    print(f"Weights for alpha={alpha} : {weights.flatten()}, T = {T}")

best_weights = weights_list[0]
best_T = T_list[0]

plt.figure(figsize=(10, 6))
for err, alpha in zip(errors, alphas):
    plt.plot(err, label=f'eta={alpha}')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE dependence on different etas')
plt.legend()
plt.grid(True)
plt.show()

def graph(X, e, weights, T, new_points=None, new_classes=None):
    plt.figure(figsize=(8, 6))
    class0 = X[e.flatten() == 0]
    class1 = X[e.flatten() == 1]
    plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
    plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
    w1, w2 = weights.flatten()
    x1_vals = np.linspace(-7, 7, 100)
    if w2 != 0:
        x2_vals = (0.5 + T - w1 * x1_vals) / w2
        """Без 0.5 плохо будет. На лекции говорили что надо смещение чтобы оно норм показывало, иначе какашка"""
        plt.plot(x1_vals, x2_vals, color='green', label='Border (S=0.5)')
    else:
        print("w2=0, vertical line - rare situation")
    if new_points is not None and new_classes is not None:
        new_points = np.array(new_points)
        for i, cls in enumerate(new_classes):
            color = 'blue' if cls == 0 else 'red'
            plt.scatter(new_points[i, 0], new_points[i, 1], color=color, marker='x', s=100)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Data, border line and other dots')
    plt.legend()
    plt.grid(True)
    plt.show()

graph(X, e, best_weights, best_T)

def new_point(x1, x2, weights, T, threshold=0):
    y = x1 * weights[0] + x2 * weights[1] - T
    y = (y * 2) - 1 # Иначе порог должен быть 0.5, чтобы работало норм. 
    return y, (1 if y > threshold else 0)

new_points = []
new_classes = []
print("\nFunctionality mod: enter 2 value in [-6, 6] or 'exit'.")
while True:
    try:
        input_str = input("Enter x1 and x2: ").strip()
        if input_str.lower() == 'exit':
            break
        x1, x2 = map(float, input_str.split())
        if not (-6 <= x1 <= 6 and -6 <= x2 <= 6):
            print("Caution: inputs not in [-6, 6]. There can be anomalies")
        pred, cls = new_point(x1, x2, best_weights, best_T)
        print(f"Point ({x1}, {x2}): prediction {pred}, class {cls}")
        new_points.append([x1, x2])
        new_classes.append(cls)
        graph(X, e, best_weights, best_T, new_points, new_classes)
    except ValueError:
        print("Error.")

