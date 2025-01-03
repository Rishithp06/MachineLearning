# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Ensure the use of interactive plots (for Jupyter Notebooks)
# %matplotlib widget  # Uncomment this if running in a Jupyter environment

# Custom styles for better visuals (optional)
plt.style.use('./deeplearning.mplstyle')  # Use this style if available

# --- Step 1: Define the sigmoid function ---
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    # Apply the sigmoid formula
    g = 1 / (1 + np.exp(-z))
    return g

# --- Step 2: Test the sigmoid function with different inputs ---
# Generate an array of evenly spaced values between -10 and 10
z = np.arange(-10, 11)

# Compute sigmoid values for the array
y = sigmoid(z)

# Pretty print the input and output arrays
print("Input (z), Output (sigmoid(z))")
print(np.c_[z, y])  # Combine and print columns for z and sigmoid(z)

# --- Step 3: Visualize the sigmoid function ---
# Create a plot for the sigmoid function
plt.figure(figsize=(6, 4))  # Set figure size
plt.plot(z, y, c="b", label="Sigmoid Function")  # Plot sigmoid curve
plt.axhline(y=0.5, color="orange", linestyle="--", label="y=0.5 Threshold")  # Add a threshold line
plt.title("Sigmoid Function")  # Set title
plt.xlabel("z")  # Label x-axis
plt.ylabel("sigmoid(z)")  # Label y-axis
plt.legend()  # Add a legend
plt.grid(True)  # Add grid lines
plt.show()  # Display the plot

# --- Step 4: Logistic Regression Setup ---
# Training data (tumor sizes and labels)
x_train = np.array([0., 1, 2, 3, 4, 5])  # Input feature: Tumor sizes
y_train = np.array([0, 0, 0, 1, 1, 1])  # Labels: 0 = benign, 1 = malignant

# Initialize parameters for the logistic regression model
w = np.zeros((1))  # Weight (start with 0)
b = 0              # Bias (start with 0)

# --- Step 5: Define logistic regression predictions ---
def logistic_regression(x, w, b):
    """
    Perform logistic regression by applying the sigmoid function
    to the linear combination of inputs.

    Args:
        x (ndarray): Input features (e.g., tumor sizes).
        w (ndarray): Weight (parameter).
        b (float): Bias (parameter).

    Returns:
        predictions (ndarray): Probabilities for each input x.
    """
    # Compute z = wx + b (linear combination)
    z = w * x + b
    # Apply the sigmoid function to z
    predictions = sigmoid(z)
    return predictions, z

# --- Step 6: Visualize Logistic Regression ---
# Compute predictions and z-values
predictions, z_values = logistic_regression(x_train, w, b)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color="red", label="Training Data")  # Plot training data points
plt.plot(x_train, predictions, label="Sigmoid Curve (Logistic Regression)", color="blue")  # Sigmoid curve
plt.plot(x_train, z_values, label="z = wx + b (Linear Component)", color="orange", linestyle="--")  # Linear part
plt.axhline(y=0.5, color="green", linestyle="--", label="Threshold (y=0.5)")  # Decision boundary
plt.title("Logistic Regression")  # Title of the plot
plt.xlabel("Tumor Size (x)")  # X-axis label
plt.ylabel("Predicted Probability (y)")  # Y-axis label
plt.legend()  # Add a legend
plt.grid(True)  # Add grid
plt.show()  # Display the plot

# --- Step 7: Add a New Data Point (Optional Interaction) ---
# Manually add a new point for testing robustness
new_point = np.array([10])  # Large tumor size
x_train = np.append(x_train, new_point)  # Append new point to training data
y_train = np.append(y_train, [1])  # Append label (malignant)

# Recompute predictions with updated data
predictions, z_values = logistic_regression(x_train, w, b)

# Plot updated results
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color="red", label="Updated Training Data")  # Updated training data
plt.plot(x_train, predictions, label="Updated Sigmoid Curve", color="blue")  # Updated sigmoid curve
plt.plot(x_train, z_values, label="Updated z = wx + b", color="orange", linestyle="--")  # Updated linear part
plt.axhline(y=0.5, color="green", linestyle="--", label="Threshold (y=0.5)")  # Decision boundary
plt.title("Logistic Regression with Additional Data Point")  # Title
plt.xlabel("Tumor Size (x)")  # X-axis label
plt.ylabel("Predicted Probability (y)")  # Y-axis label
plt.legend()  # Add legend
plt.grid(True)  # Add grid
plt.show()  # Display the updated plot

# --- Step 8: Congratulations! ---
print("Congratulations! You have successfully implemented and explored the sigmoid function and logistic regression.")
