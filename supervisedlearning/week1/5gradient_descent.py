# Function to perform gradient descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to optimize w and b.
    Args:
        x (ndarray): Features (input data).
        y (ndarray): Target values (output data).
        w_in (float): Initial weight.
        b_in (float): Initial bias.
        alpha (float): Learning rate.
        num_iters (int): Number of iterations.
        cost_function (function): Function to compute cost.
        gradient_function (function): Function to compute gradients.
    Returns:
        tuple: Final parameters (w, b) and history of cost and parameters.
    """
    J_history = []  # History of cost values
    p_history = []  # History of parameter values
    w = w_in  # Initialize weight
    b = b_in  # Initialize bias
    
    # Perform gradient descent
    for i in range(num_iters):
        # Compute gradients
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        # Update parameters
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        # Save cost and parameters for analysis
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        
        # Print progress at intervals
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}, dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}, w: {w: 0.3e}, b: {b: 0.5e}")
    
    return w, b, J_history, p_history
