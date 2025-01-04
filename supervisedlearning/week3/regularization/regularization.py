import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Regularized cost function
def compute_cost_with_regularization(X, y, w, b, lambda_):
    m = X.shape[0]
    cost = 0.0
    reg_cost = 0.0
    
    # Compute standard cost
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    
    cost /= m
    
    # Compute regularization term
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    
    # Total cost
    total_cost = cost + reg_cost
    return total_cost

# Example dataset
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w = np.zeros(X_train.shape[1])
b = 0
lambda_ = 0.1

# Compute cost with regularization
cost = compute_cost_with_regularization(X_train, y_train, w, b, lambda_)
print(f"Regularized cost: {cost}")
