import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Regularized cost function
def compute_cost_with_regularization(X, y, w, b, lambda_):
    m = X.shape[0]
    cost = 0.0
    
    # Compute the standard cost
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost /= m
    
    # Add regularization term
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost

# Gradient computation with regularization
def compute_gradient_with_regularization(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.0
    
    # Compute gradients
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        error = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    
    dj_dw = dj_dw / m + (lambda_ / m) * w  # Regularization added here
    dj_db = dj_db / m  # Bias not regularized
    return dj_dw, dj_db

# Gradient descent with regularization
def gradient_descent_with_regularization(X, y, w, b, alpha, num_iters, lambda_):
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_with_regularization(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i % 100 == 0:  # Print cost every 100 iterations
            cost = compute_cost_with_regularization(X, y, w, b, lambda_)
            print(f"Iteration {i}: Cost {cost}")
    return w, b

# Example dataset
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Initialize parameters
w = np.zeros(X_train.shape[1])
b = 0.0
alpha = 0.1
num_iters = 1000
lambda_ = 1.0

# Run gradient descent with regularization
w, b = gradient_descent_with_regularization(X_train, y_train, w, b, alpha, num_iters, lambda_)
print(f"Trained parameters: w = {w}, b = {b}")
