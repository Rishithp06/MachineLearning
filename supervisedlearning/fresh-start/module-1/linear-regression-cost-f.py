import numpy as np
import matplotlib.pyplot as plt

# Dataset
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 5, 4])

# Cost function
def compute_cost(x, y, w, b):
    m = len(x)
    predictions = w * x + b
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Test different w, b values
params = [(0, 1.5), (0.5, 0), (1, 1)]
for w, b in params:
    cost = compute_cost(x, y, w, b)
    print(f"w={w}, b={b}, Cost={cost:.4f}")
    
    # Plot data and line
    plt.scatter(x, y, color='blue', label='Data')
    x_line = np.array([0, 5])
    y_line = w * x_line + b
    plt.plot(x_line, y_line, label=f'w={w}, b={b}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Line Fit (Cost={cost:.4f})')
    plt.legend()
    plt.show()