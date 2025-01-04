import numpy as np
from sklearn.linear_model import LogisticRegression

# Step 1: Dataset
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# Step 2: Initialize and Train the Model
lr_model = LogisticRegression()
lr_model.fit(X, y)

# Step 3: Make Predictions
y_pred = lr_model.predict(X)
print("Predictions on training set:", y_pred)

# Step 4: Calculate Accuracy
accuracy = lr_model.score(X, y)
print("Accuracy on training set:", accuracy)