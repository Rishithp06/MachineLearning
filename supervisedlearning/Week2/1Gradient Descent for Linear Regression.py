Purpose: Optimize parameters w and b to minimize the cost function.

import numpy as np

# Simulated data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature (size)
y = np.array([2, 4, 6, 8, 10])               # Target (price)

# Initialize parameters
m, n = X.shape  # m = number of examples, n = number of features
W = np.zeros(n)  # Initialize weights
b = 0            # Initialize bias
alpha = 0.01     # Learning rate
iterations = 100 # Number of iterations

# Gradient descent
for _ in range(iterations):
    # Compute predictions
    predictions = np.dot(X, W) + b

    # Compute gradients
    errors = predictions - y
    dW = (1 / m) * np.dot(X.T, errors)
    db = (1 / m) * np.sum(errors)

    # Update parameters
    W -= alpha * dW
    b -= alpha * db

print(f"Trained weights: {W}, Trained bias: {b}")
