

Purpose: Fit curves (non-linear relationships) to the data.

Manual Implementation
# Simulated data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature (size)
y = np.array([2, 5, 10, 17, 26])             # Target (price)

# Polynomial features
X_poly = np.hstack([X, X**2, X**3])  # Add x^2 and x^3

# Scale features
X_poly_scaled = (X_poly - np.mean(X_poly, axis=0)) / np.std(X_poly, axis=0)

# Train linear regression model
W = np.linalg.inv(X_poly_scaled.T @ X_poly_scaled) @ X_poly_scaled.T @ y

# Predictions
y_pred = X_poly_scaled @ W

# Plot results
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Polynomial Fit')
plt.legend()
plt.show()




Using Scikit-learn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Polynomial regression pipeline
model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('linear_regression', LinearRegression())
])

# Train the model
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plot results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Polynomial Fit')
plt.legend()
plt.show()
