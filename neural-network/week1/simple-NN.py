import numpy as np

# Sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward pass function
def neural_network(x, W1, b1, W2, b2, W3, b3, W4, b4):
    # Hidden Layer 1
    z1 = np.dot(W1, x) + b1  # (100,)
    a1 = sigmoid(z1)

    # Hidden Layer 2
    z2 = np.dot(W2, a1) + b2  # (50,)
    a2 = sigmoid(z2)

    # Hidden Layer 3
    z3 = np.dot(W3, a2) + b3  # (20,)
    a3 = sigmoid(z3)

    # Output Layer
    z4 = np.dot(W4, a3) + b4  # scalar
    a4 = sigmoid(z4)

    return a4, a1, a2, a3

# Data setup
num_pixels = 784
data = np.random.rand(100, num_pixels)        # 100 samples of 28x28
labels = np.random.randint(0, 2, 100)         # Binary classification

# Weight initialization
W1 = np.random.randn(100, num_pixels) * 0.01
b1 = np.zeros(100)

W2 = np.random.randn(50, 100) * 0.01
b2 = np.zeros(50)

W3 = np.random.randn(20, 50) * 0.01
b3 = np.zeros(20)

W4 = np.random.randn(1, 20) * 0.01
b4 = 0.0

# Hyperparameters
learning_rate = 0.1
epochs = 1000

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(data)):
        x = data[i]
        y = labels[i]

        # Forward pass
        a4, a1, a2, a3 = neural_network(x, W1, b1, W2, b2, W3, b3, W4, b4)

        # Binary cross-entropy loss
        epsilon = 1e-15
        loss = -(y * np.log(a4 + epsilon) + (1 - y) * np.log(1 - a4 + epsilon))
        total_loss += loss

        # Backpropagation
        grad_a4 = (a4 - y) / (a4 * (1 - a4) + epsilon)       # scalar
        grad_W4 = grad_a4 * a3.reshape(1, -1)                # (1, 20)
        grad_b4 = grad_a4

        grad_a3 = grad_a4 * W4 * a3 * (1 - a3)               # (1, 20)
        grad_a3 = grad_a3.flatten()                         # (20,)
        grad_W3 = np.outer(grad_a3, a2)                     # (20, 50)
        grad_b3 = grad_a3

        grad_a2 = np.dot(W3.T, grad_a3) * a2 * (1 - a2)     # (50,)
        grad_W2 = np.outer(grad_a2, a1)                     # (50, 100)
        grad_b2 = grad_a2

        grad_a1 = np.dot(W2.T, grad_a2) * a1 * (1 - a1)     # (100,)
        grad_W1 = np.outer(grad_a1, x)                      # (100, 784)
        grad_b1 = grad_a1

        # Update weights and biases
        W4 -= learning_rate * grad_W4
        b4 -= learning_rate * grad_b4
        W3 -= learning_rate * grad_W3
        b3 -= learning_rate * grad_b3
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch}, Loss: {avg_loss[0]:.4f}")

# Test prediction
x_test = np.random.rand(num_pixels)
prob, _, _, _ = neural_network(x_test, W1, b1, W2, b2, W3, b3, W4, b4)
print(f"Probability of being Alice: {prob[0]:.3f}")
