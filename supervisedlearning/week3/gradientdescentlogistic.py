import numpy as np
import matplotlib.pyplot as plt
import copy, math
from lab_utils_common import dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob
plt.style.use('./deeplearning.mplstyle')

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute Gradient for Logistic Regression
def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for logistic regression.

    Args:
      X (ndarray (m, n)): Data, m examples with n features.
      y (ndarray (m,)): Target values.
      w (ndarray (n,)): Model parameters.
      b (scalar): Model parameter.

    Returns:
      dj_dw (ndarray (n,)): Gradient of the cost w.r.t. w.
      dj_db (scalar): Gradient of the cost w.r.t. b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

# Gradient Descent Implementation
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent.

    Args:
      X (ndarray (m, n)): Data, m examples with n features.
      y (ndarray (m,)): Target values.
      w_in (ndarray (n,)): Initial values of model parameters.
      b_in (scalar): Initial values of model parameter.
      alpha (float): Learning rate.
      num_iters (int): Number of iterations.

    Returns:
      w (ndarray (n,)): Updated values of parameters.
      b (scalar): Updated value of parameter.
      J_history (list): Cost at each iteration.
    """
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost_logistic(X, y, w, b))
        
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]}")

    return w, b, J_history

# Dataset: Two-Variable Data
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Initialize Parameters
w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.0
alpha = 0.1
num_iters = 10000

# Run Gradient Descent
w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alpha, num_iters)
print(f"Updated parameters: w = {w_out}, b = {b_out}")

# Plot Probability and Decision Boundary
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt_prob(ax, w_out, b_out)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
ax.axis([0, 4, 0, 3.5])
plot_data(X_train, y_train, ax)

# Plot Decision Boundary
x0 = -b_out / w_out[0]
x1 = -b_out / w_out[1]
ax.plot([0, x0], [x1, 0], c=dlc["dlblue"], lw=1)
plt.show()

# Dataset: One-Variable Data
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

# Visualize One-Variable Data
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plt_tumor_data(x_train, y_train, ax)
plt.show()

# Contour Plot for One-Variable Data
w_range = np.array([-1, 7])
b_range = np.array([1, -14])
plt_quad_logistic(x_train, y_train, w_range, b_range)