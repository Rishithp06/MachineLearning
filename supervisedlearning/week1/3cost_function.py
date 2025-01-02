# Function to calculate the cost
def compute_cost(x, y, w, b):
    """
    Compute the cost J(w, b) for linear regression.
    Args:
        x (ndarray): Features (input data).
        y (ndarray): Target values (output data).
        w (float): Weight parameter.
        b (float): Bias parameter.
    Returns:
        float: The cost value.
    """
    m = x.shape[0]  # Number of training examples
    cost = 0  # Initialize cost
    
    # Calculate the squared error for each example
    for i in range(m):
        f_wb = w * x[i] + b  # Prediction
        cost += (f_wb - y[i])**2  # Squared error

    # Average the cost over all examples
    total_cost = 1 / (2 * m) * cost
    return total_cost
