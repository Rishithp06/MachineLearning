def logistic_cost(y_true, y_pred):
    """
    Compute the logistic cost over the entire dataset.

    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted probabilities.

    Returns:
        cost (float): Average logistic cost.
    """
    m = y_true.shape[0]  # Number of examples
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



FULL CODEEEE : 

    
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic loss for a single example
def logistic_loss(y_true, y_pred):
    """
    Compute the logistic loss for a single example.

    Args:
        y_true (float): True label (0 or 1).
        y_pred (float): Predicted probability (output of sigmoid).

    Returns:
        loss (float): Logistic loss.
    """
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

# Logistic cost function for a dataset
def logistic_cost(y_true, y_pred):
    """
    Compute the logistic cost over the entire dataset.

    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted probabilities.

    Returns:
        cost (float): Average logistic cost.
    """
    m = y_true.shape[0]  # Number of examples
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example dataset
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Model parameters
w, b = 1, -2

# Predictions
z = w * x_train + b
y_pred = sigmoid(z)

# Compute cost
cost = logistic_cost(y_train, y_pred)
print("Logistic Cost:", cost)

# Compute loss for a single example
single_loss = logistic_loss(y_train[0], y_pred[0])
print("Logistic Loss for first example:", single_loss)
