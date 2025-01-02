# Function to compute gradients
def compute_gradient(x, y, w, b):
    """
    Computes the gradient of the cost function w.r.t. parameters w and b.
    Args:
        x (ndarray): Features (input data).
        y (ndarray): Target values (output data).
        w (float): Current weight parameter.
        b (float): Current bias parameter.
    Returns:
        tuple: Gradients (dj_dw, dj_db).
    """
    m = x.shape[0]  # Number of training examples
    dj_dw = 0  # Initialize gradient w.r.t. w
    dj_db = 0  # Initialize gradient w.r.t. b
    
    # Compute gradients for each example
    for i in range(m):
        f_wb = w * x[i] + b  # Prediction
        dj_dw += (f_wb - y[i]) * x[i]  # Gradient w.r.t. w
        dj_db += (f_wb - y[i])         # Gradient w.r.t. b

    # Average gradients over all examples
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
