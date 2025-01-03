import numpy as np

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
