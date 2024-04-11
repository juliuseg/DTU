# Code written by Julius Gronager, s204427
# Code written in collaboration with Chat-gpt4

from LoadData import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your data here
# X, y_binary = LoadData() # Make sure you load your data correctly

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce to one principal component
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Train logistic regression on the PCA-transformed data
optimal_C = 1  # Use the average optimal C obtained from previous cross-validation
model = LogisticRegression(C=optimal_C, max_iter=1000)
model.fit(X_pca, y_binary)

# Plot the PCA-transformed points
plt.scatter(X_pca[:100, 0], y_binary[:100], color='black', zorder=20, s=5)

# Plot the logistic regression decision function
x_values = np.linspace(X_pca.min(), X_pca.max(), 300)
# Calculate the probability estimates
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]

plt.plot(x_values, y_values, color='red', linewidth=2)

# Additional plot settings
plt.xlabel('Principal Component')
plt.ylabel('Probability')
plt.title('Logistic Regression on Principal Component')
plt.axhline(.5, color='.5')
plt.ylabel('y')
plt.xlabel('X')
plt.xticks(())
plt.yticks(())
plt.ylim(-0.25, 1.25)
plt.xlim(X_pca.min(), X_pca.max())
plt.show()
