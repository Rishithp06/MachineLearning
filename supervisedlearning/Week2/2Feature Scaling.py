Purpose: Standardize feature ranges to improve gradient descent performance.
Normalization

X = np.array([[300, 0], [2000, 5]])  # Features with different ranges
X_max = np.max(X, axis=0)            # Maximum values of each column
X_scaled = X / X_max                 # Normalize to [0, 1]
print("Normalized Features:\n", X_scaled)


Mean Normalization
X_mean = np.mean(X, axis=0)          # Mean of each column
X_min = np.min(X, axis=0)            # Minimum values of each column
X_max = np.max(X, axis=0)            # Maximum values of each column
X_scaled = (X - X_mean) / (X_max - X_min)
print("Mean Normalized Features:\n", X_scaled)



Z-Score Normalization
X_mean = np.mean(X, axis=0)          # Mean of each column
X_std = np.std(X, axis=0)            # Standard deviation of each column
X_scaled = (X - X_mean) / X_std
print("Z-Score Normalized Features:\n", X_scaled)
