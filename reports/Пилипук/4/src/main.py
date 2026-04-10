import numpy as np
import matplotlib.pyplot as plt

N = 7
X_raw = np.array([[int(ch) for ch in f'{i:07b}'] for i in range(2**N)], dtype=float)
mask = np.array([0, 0, 0, 1, 1, 0, 0])

def solve(X, mask):
    masked_bits = X * mask
    return (np.sum(masked_bits, axis=1) > 0).astype(float).reshape(-1, 1)

e = solve(X_raw, mask)

split = int(len(X_raw) * 0.67)
X_train, X_test = X_raw[:split], X_raw[split:]
e_train, e_test = e[:split], e[split:]

def predict(weights, X, T):
    return X @ weights - T

def MSE(e, y):
    return np.mean((y - e) ** 2)

def sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))

def trainMSE(X, e, alpha, adaptive=False, epochs=10000, tol=1e-5):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    T = 0.0
    errors = []

    current_alpha = alpha

    for epoch in range(epochs):
        y = X @ weights - T
        a = sigmoid(y) 
    
        error = np.mean((a - e) ** 2)
        errors.append(error)

        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < tol:
            print(f"MSE minimized at epoch = {epoch}") 
            break
        
        delta = (a - e) * (a * (1 - a)) 

        grad_w = (X.T @ delta) / n_samples
        grad_T = np.mean(delta)

        weights -= current_alpha * grad_w
        T += current_alpha * grad_T 

        if adaptive: 
            current_alpha = alpha / (1 + 0.001 * epoch)

    return weights, T, errors

weights_fixed, T_fixed, err_fixed = trainMSE(X_raw, e, alpha=0.1, adaptive=False)
weights_adapt, T_adapt, err_adapt = trainMSE(X_raw, e, alpha=0.1, adaptive=True)

plt.figure(figsize=(10, 6))
plt.plot(err_fixed, label="MSE α=0.1 (Fixed)")
plt.plot(err_adapt, label="MSE Adaptive")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Convergence comparison")
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Final Weights (Fixed): {weights_fixed.flatten()}")
print(f"Final T (Fixed): {T_fixed}")

print(f"Final Weights (Adapt): {weights_adapt.flatten()}")
print(f"Final T (Adapt): {T_adapt}")

for i in range(len(X_test)):
    y = sigmoid(predict(weights_fixed, X_test[i], T_fixed))
    num = 1 if y >= 0.5 else 0
    print(f"Ahmmf... Im thinking that this is {num} with {y}%. Actually this is - {e_test[i]}")