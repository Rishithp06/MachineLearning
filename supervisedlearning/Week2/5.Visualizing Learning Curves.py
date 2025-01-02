Purpose: Check if gradient descent is converging.
# Simulated data
iterations = list(range(1, 501))
cost_values = [1 / (i + 1) for i in iterations]  # Simulated cost values

# Plotting the learning curve
plt.plot(iterations, cost_values)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost J(w, b)")
plt.title("Learning Curve")
plt.show()
