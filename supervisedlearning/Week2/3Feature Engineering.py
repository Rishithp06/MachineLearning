Create new features to capture meaningful relationships in the data.


# Original features
X = np.array([[10, 20], [30, 40], [50, 60]])  # Example: Width and Depth of lots

# Create a new feature: Area = Width * Depth
X_area = X[:, 0] * X[:, 1]
X_new = np.hstack((X, X_area.reshape(-1, 1)))  # Combine with original features

print("Features after Engineering (Width, Depth, Area):\n", X_new)
