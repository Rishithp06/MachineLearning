import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and preprocess MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

# Define the model
model = models.Sequential([
    layers.Dense(25, activation='relu', input_shape=(784,)),  # Hidden Layer 1
    layers.Dense(15, activation='relu'),                     # Hidden Layer 2
    layers.Dense(10)                                        # Output Layer (Linear)
])

# Compile the model
model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")