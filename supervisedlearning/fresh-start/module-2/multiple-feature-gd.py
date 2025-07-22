import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([
    [1000, 2, 1, 20],   # Example 1
    [1416, 3, 2, 40],   # Example 2
    [2000, 4, 2, 10]    # Example 3
])
y = np.array([150, 173.6, 230])  # Prices in thousands
m, n = X.shape  # m=3 examples, n=4 features

# Initialize parameters
w = np.array([0.1, 4, 10, -2])  # From transcript
b = 80

# Cost function
def calculate_cost(X, y, w, b):
    predictions = np.dot(X, w) + b  # Vectorized: Xw + b
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

# Gradients
def calculate_gradients(X, y, w, b):
    predictions = np.dot(X, w) + b
    errors = predictions - y
    grad_w = (1 / m) * np.dot(X.T, errors)  # Vectorized gradient for w
    grad_b = (1 / m) * np.sum(errors)
    return grad_w, grad_b

# Gradient descent
def run_gradient_descent(X, y, w_init, b_init, alpha, steps):
    w, b = w_init, b_init
    cost_history = []
    for _ in range(steps):
        cost = calculate_cost(X, y, w, b)
        cost_history.append(cost)
        grad_w, grad_b = calculate_gradients(X, y, w, b)
        w = w - alpha * grad_w
        b = b - alpha * grad_b
    return w, b, cost_history

# Run gradient descent
alpha = 1e-5  # Small learning rate due to large feature values
steps = 1000
w_init = np.zeros(n)  # Start with w=[0,0,0,0]
b_init = 0
w, b, cost_history = run_gradient_descent(X, y, w_init, b_init, alpha, steps)

print(f"Final w: {w}, b: {b:.2f}, Cost: {cost_history[-1]:.2f}")

# Plot cost history
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J(w,b)')
plt.title('Cost vs. Iteration (Multiple Features)')
plt.show()

# Predictions
predictions = np.dot(X, w) + b
print("Predictions:", predictions)
print("Actual:", y)

# Simple evaluation (MSE)
mse = np.mean((predictions - y) ** 2)
print(f"MSE: {mse:.2f}")