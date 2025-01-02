# Plot cost vs. iterations
We visualize the cost function's behavior over iterations.
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])  # Initial iterations
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])  # Later iterations
ax1.set_title("Cost vs. iteration (start)"); ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost'); ax2.set_ylabel('Cost')
ax1.set_xlabel('Iteration step'); ax2.set_xlabel('Iteration step')
plt.show()




# Contour plot with gradient descent path
The contour plot shows how the gradient descent algorithm progresses.
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)




# Demonstrating divergence
We demonstrate how a high learning rate can cause divergence.
w_init = 0
b_init = 0
alpha_high = 0.8  # High learning rate
iterations_high = 10

w_final_div, b_final_div, J_hist_div, p_hist_div = gradient_descent(
    x_train, y_train, w_init, b_init, alpha_high, iterations_high, compute_cost, compute_gradient)

plt_divergence(p_hist_div, J_hist_div, x_train, y_train)
plt.show()
