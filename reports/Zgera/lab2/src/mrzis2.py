import numpy as np
import matplotlib.pyplot as plt

data_points = np.array([
    [ 2.0,  6.0],
    [-2.0,  6.0],
    [ 2.0, -6.0],
    [-2.0, -6.0]
])

targets = np.array([0.0, 1.0, 0.0, 0.0])

target_error = 0.01
max_iterations = 200

def activation(weights, bias, input_vector):
    return np.dot(weights, input_vector) - bias

def fixed_lr_training(data, targets, learning_rate, error_limit, max_iters):
    weights = np.random.normal(size=data.shape[1])
    bias = np.random.normal()
    error_progress = []
    
    for iteration in range(max_iters):
        for sample, target in zip(data, targets):
            output = activation(weights, bias, sample)
            delta = target - output
            weights += learning_rate * delta * sample
            bias -= learning_rate * delta
        
        current_error = sum((targets[idx] - activation(weights, bias, data[idx]))**2
                           for idx in range(len(data)))
        error_progress.append(current_error)
        
        if current_error <= error_limit:
            break
            
    return weights, bias, error_progress

def adaptive_lr_training(data, targets, error_limit, max_iters):
    weights = np.random.normal(size=data.shape[1])
    bias = np.random.normal()
    error_progress = []
    
    for iteration in range(max_iters):
        for sample, target in zip(data, targets):
            output = activation(weights, bias, sample)
            delta = target - output
            adaptive_rate = 1.0 / (1.0 + np.dot(sample, sample))
            weights += adaptive_rate * delta * sample
            bias -= adaptive_rate * delta
        
        current_error = sum((targets[idx] - activation(weights, bias, data[idx]))**2
                           for idx in range(len(data)))
        error_progress.append(current_error)
        
        if current_error <= error_limit:
            break
            
    return weights, bias, error_progress

fixed_rate = 0.01

weights_fixed, bias_fixed, progress_fixed = fixed_lr_training(data_points, targets, fixed_rate, target_error, max_iterations)
weights_adapt, bias_adapt, progress_adapt = adaptive_lr_training(data_points, targets, target_error, max_iterations)

print(f"Fixed rate training: iterations = {len(progress_fixed)}")
print(f"Adaptive rate training: iterations = {len(progress_adapt)}")
print(f"Final error (fixed):   {progress_fixed[-1]:.6f}")
print(f"Final error (adaptive): {progress_adapt[-1]:.6f}")

w1, w2 = weights_adapt
theta = bias_adapt

print("\nDecision boundary equation (adaptive method):")
print(f"  {w1:.4f} * x₁ + {w2:.4f} * x₂ - {theta:.4f} = 0")

if abs(w2) > 1e-8:
    m = -w1 / w2
    b = theta / w2
    print(f"  y = {m:.4f} * x₁ + {b:.4f}")
else:
    if abs(w1) > 1e-8:
        x_const = theta / w1
        print(f"  x₁ = {x_const:.4f}  (vertical line)")
    else:
        print("  Undefined line (w1 ≈ w2 ≈ 0)")

plt.figure(figsize=(10, 6))
plt.plot(progress_fixed, color="purple", label="Fixed learning rate")
plt.plot(progress_adapt, color="orange", label="Adaptive learning rate")
plt.xlabel("Iterations (epochs)")
plt.ylabel("Total squared error Σ(e - y)²")
plt.title("Training Error Progression")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(7, 7))
for idx in range(len(data_points)):
    marker_color = "red" if targets[idx] == 1 else "blue"
    plt.scatter(data_points[idx, 0], data_points[idx, 1], color=marker_color, s=100, edgecolor='black')

x_range = np.linspace(-7, 7, 300)
if abs(weights_adapt[1]) > 1e-10:
    y_range = (bias_adapt - weights_adapt[0] * x_range) / weights_adapt[1]
    plt.plot(x_range, y_range, 'k', linewidth=2, label='Decision boundary')
else:
    x_const = bias_adapt / weights_adapt[0] if abs(weights_adapt[0]) > 1e-10 else 0
    plt.axvline(x=x_const, color='k', linewidth=2, label='Decision boundary')

plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.grid(True)
plt.title("Decision Boundary (Adaptive Training)")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.show()

def perform_classification(coord_x, coord_y, weights, bias):
    computed_act = weights[0] * coord_x + weights[1] * coord_y - bias
    predicted_class = 1 if computed_act > 0 else 0
    
    print(f"\nPoint:          ({coord_x}, {coord_y})")
    print(f"Activation:     {computed_act:.4f}")
    print(f"Predicted class: {predicted_class}")
    
    plt.figure(figsize=(7, 7))
    for idx in range(len(data_points)):
        marker_color = "red" if targets[idx] == 1 else "blue"
        plt.scatter(data_points[idx, 0], data_points[idx, 1], color=marker_color, s=100, edgecolor='black')
    
    if abs(weights[1]) > 1e-10:
        x_range = np.linspace(-7, 7, 300)
        y_range = (bias - weights[0] * x_range) / weights[1]
        plt.plot(x_range, y_range, 'k', linewidth=2)
    else:
        x_const = bias / weights[0] if abs(weights[0]) > 1e-10 else 0
        plt.axvline(x=x_const, color='k', linewidth=2)
    
    plt.scatter(coord_x, coord_y, color="green", s=180, marker="x", linewidths=3)
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.grid(True)
    plt.title("User Point Classification")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.show()

while True:
    print("\nEnter point coordinates (or 'exit' / 'q' to finish):")
    input_x = input("x1 = ")
    if input_x.lower() in ['exit', 'q']:
        break
    input_y = input("x2 = ")
    if input_y.lower() in ['exit', 'q']:
        break
    try:
        perform_classification(float(input_x), float(input_y), weights_adapt, bias_adapt)
    except ValueError:
        print("Error: please enter numeric values!\n")