import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')

# Training data
x_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.longdouble)
y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.longdouble)

# Visualize the data
plt_simple_example(x_train, y_train)

# Visualize squared error cost surface
plt.close('all')
plt_logistic_squared_error(x_train, y_train)
plt.show()

# Visualize logistic loss curves
plt_two_logistic_loss_curves()
plt.show()

# Visualize logistic cost function
plt.close('all')
cst = plt_logistic_cost(x_train, y_train)
plt.show()
