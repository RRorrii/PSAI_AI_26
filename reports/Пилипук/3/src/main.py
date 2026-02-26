import numpy as np
import matplotlib.pyplot as plt

X = np.array([[6, 1], [-6, 1], [6, -1], [-6, -1]])
e = np.array([0, 0, 0, 1]).reshape(-1, 1)

def predict(weights, X, T):
    return X @ weights - T

def MSE(e, y):
    return np.mean((y - e) ** 2)

def BCE(e, y):
    return -np.mean(e * np.log(y) + (1 - e) * np.log(1 - y))

def sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))

def trainMSE(X, e, alpha, adaptive=False, epochs=10000, tol=1e-6):
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

        if adaptive: alpha = 1 / (1 + np.mean((X) ** 2))

    return weights, T, errors

def trainBCE(X, e, alpha, adaptive=False, epochs=10000, tol=1e-9):
    N = X.shape[0]
    weights = np.zeros((X.shape[1], 1))
    T = 0.0
    errors = []

    for epoch in range(epochs):
        y = predict(weights, X, T)
        error = BCE(e, sigmoid(y))
        errors.append(error)

        if len(errors) > 1 and abs(errors[-1] - errors[-2]) < tol:
            print(f"BCE minimized at epoch = {epoch}")
            break

        grad_w = (X.T @ (y - e)) / N
        grad_T = np.mean(y - e)

        weights -= alpha * grad_w
        T += alpha * grad_T

        if adaptive: alpha = 1 / (1 + np.mean((X) ** 2))

    return weights, T, errors

errorsMSE = []
errorsBCE = []
weightsMSE = []
weightsBCE = []
TMSE = []
TBCE = []

experiments = [
    ("MSE",    trainMSE,  False, weightsMSE, TMSE, errorsMSE, "a=0.02"),
    ("MSE",    trainMSE,  True,  weightsMSE, TMSE, errorsMSE, "adaptive"),
    ("BCE",    trainBCE,  False, weightsBCE, TBCE, errorsBCE, "a=0.02"),
    ("BCE",    trainBCE,  True,  weightsBCE, TBCE, errorsBCE, "adaptive"),
]

for loss_name, func, adaptive, w_list, t_list, err_list, label in experiments:
    weights, T, error = func(X, e, 0.02, adaptive=adaptive)
    
    w_list.append(weights)
    t_list.append(T)
    err_list.append(error)
    
    print(f"Weights for {loss_name} {label}: {weights.flatten()}, T = {T}")

plt.figure(figsize=(10, 6))

labels = ["MSE α=0.02", "MSE adaptive", "BCE a=0.02", "BCE adaptive"]
error_lists = [errorsMSE[0], errorsMSE[1], errorsBCE[0], errorsBCE[1]]

for err, lbl in zip(error_lists, labels):
    if len(err) > 0:
        epochs = np.arange(1, len(err) + 1)
        plt.plot(epochs, err, label=lbl, linewidth=1.8)

plt.xlabel("Epoch")
plt.ylabel("Error value")
plt.title("Convergence comparison: MSE vs BCE, fixed vs adaptive step")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

def graph(
    X, e,
    weights_mse, T_mse,         
    weights_bce, T_bce,           
    new_points=None,
    new_classes=None,
    figsize=(9, 7),
    title="Data points and decision boundaries (MSE vs BCE)"
):

    fig, ax = plt.subplots(figsize=figsize)

    class0 = X[e.flatten() == 0]
    class1 = X[e.flatten() == 1]

    ax.scatter(class0[:, 0], class0[:, 1],
               color='blue', s=60, alpha=0.75, label='Class 0')
    ax.scatter(class1[:, 0], class1[:, 1],
               color='red',  s=60, alpha=0.75, label='Class 1')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1 = np.linspace(x_min, x_max, 200)

    w1_m, w2_m = weights_mse.flatten()
    if abs(w2_m) > 1e-8:
        x2_mse = (0.5 + T_mse - w1_m * x1) / w2_m
        ax.plot(x1, x2_mse, color='green', lw=2.2,
                label='MSE decision boundary (a=0.02)')
    else:
        print("MSE: w2 ≈ 0 → vertical line (rare case)")

    w1_b, w2_b = weights_bce.flatten()
    if abs(w2_b) > 1e-8:
        x2_bce = (0.5 + T_bce - w1_b * x1) / w2_b
        ax.plot(x1, x2_bce, color='darkorange', lw=2.2, linestyle='--',
                label='BCE decision boundary (a=0.02)')
    else:
        print("BCE: w2 ≈ 0 → vertical line (rare case)")

    if new_points is not None and new_classes is not None:
        new_points = np.asarray(new_points)
        for i, cls in enumerate(new_classes):
            color = 'blue' if cls == 0 else 'red'
            ax.scatter(new_points[i, 0], new_points[i, 1],
                       color=color, marker='x', s=144, lw=2.5,
                       label='New point' if i == 0 else None)

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

best_weights_mse = weightsMSE[0]
best_T_mse = TMSE[0]

best_weights_bce = weightsBCE[0]
best_T_bce = TBCE[0]

graph(X, e, best_weights_mse, best_T_mse, best_weights_bce, best_T_bce)

def new_point(x1, x2, weights, T, threshold=0):
    y = x1 * weights[0] + x2 * weights[1] - T
    y = (y * 2) - 1
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
        pred, cls = new_point(x1, x2, best_weights_bce, best_T_bce)
        print(f"Point ({x1}, {x2}): prediction {pred}, class {cls}")
        new_points.append([x1, x2])
        new_classes.append(cls)
        graph(X, e, best_weights_bce, best_T_bce, new_points, new_classes)
    except ValueError:
        print("Error.")

