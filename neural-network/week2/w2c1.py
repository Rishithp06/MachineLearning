import tensorflow as tf
import numpy as np

# Example training data (4 images)
X = np.random.rand(4, 64).astype(np.float32) * 255  # m x 64
Y = np.array([0, 1, 1, 0], dtype=np.float32)  # m,

# Step 1: Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=25, activation='sigmoid', input_shape=(64,)),
    tf.keras.layers.Dense(units=15, activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Step 2: Compile the model
model.compile(loss='binary_crossentropy', optimizer='sgd')

# Step 3: Train the model
model.fit(X, Y, epochs=100, verbose=0)

# Inference on a new example
X_new = np.random.rand(1, 64).astype(np.float32) * 255  # 1 x 64
a3 = model.predict(X_new)
y_hat = 1 if a3[0, 0] >= 0.5 else 0

print(f"Output a_1^{[3]}: {a3[0, 0]:.4f}")
print(f"Prediction y_hat: {y_hat}")