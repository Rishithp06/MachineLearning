import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')

# Dataset
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  # (m, n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           # (m,)

# Logistic Regression Cost Function
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m
    return cost

# Example 1: Compute cost for w = [1, 1], b = -3
w_tmp = np.array([1, 1])
b_tmp = -3
print("Cost:", compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

# Decision Boundaries for b = -3 and b = -4
x0 = np.arange(0, 6)
x1 = 3 - x0
x1_other = 4 - x0

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(x0, x1, c=dlc["dlblue"], label="$b$ = -3")
ax.plot(x0, x1_other, c=dlc["dlmagenta"], label="$b$ = -4")
plot_data(X_train, y_train, ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()

# Compare Costs for b = -3 and b = -4
w_array1 = np.array([1, 1])
b_1 = -3
w_array2 = np.array([1, 1])
b_2 = -4

print("Cost for b = -3 :", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 :", compute_cost_logistic(X_train, y_train, w_array2, b_2))
