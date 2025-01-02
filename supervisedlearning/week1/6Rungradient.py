# Initialize parameters
w_init = 0  # Initial weight
b_init = 0  # Initial bias
iterations = 10000  # Number of iterations
alpha = 0.01  # Learning rate

# Run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)

print(f"(w, b) found by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")
