import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# Cost function
def calculate_cost(w):
    predictions = w * x  # y = w * x
    errors = predictions - y
    squared_errors = errors ** 2
    return (1 / (2 * len(x))) * sum(squared_errors)

# Gradient (slope of the cost function)
def calculate_gradient(w):
    predictions = w * x
    errors = predictions - y
    return (1 / len(x)) * sum(errors * x)

# Gradient descent
w = 0  # Start at w = 0
alpha = 0.1  # Learning rate (step size)
steps = 20  # Number of steps
w_history = [w]
cost_history = [calculate_cost(w)]

for _ in range(steps):
    gradient = calculate_gradient(w)
    w = w - alpha * gradient  # Update w
    w_history.append(w)
    cost_history.append(calculate_cost(w))

print(f"Final w = {w:.2f}, Cost = {cost_history[-1]:.2f}")

# Plot J(w) vs w
w_values = np.linspace(-1, 2, 100)
costs = [calculate_cost(w) for w in w_values]
plt.plot(w_values, costs, color='blue', label='Cost Curve')
plt.scatter(w_history, cost_history, color='red', s=50, label='Gradient Descent Steps')
plt.xlabel('w')
plt.ylabel('Cost (J)')
plt.title('Gradient Descent on Cost Curve')
plt.legend()
plt.show()

# Plot final line
plt.scatter(x, y, color='blue', label='Data Points')
x_line = np.array([0, 4])
y_line = w * x_line
plt.plot(x_line, y_line, color='red', label=f'Final Line: y = {w:.2f}x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Fit')
plt.legend()
plt.show()