# Code written in collaboration with Chat-gpt4

from scipy.io import loadmat
import numpy as np

# # Load data
filename = "PowerPlant.mat"
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()

# Convert y to binary
median_y = np.median(y)
y_binary = (y > median_y).astype(int)

# Extract attribute names and dimensions
attributeNames = [name[0] for name in mat_data["attributeNames"][0]]
attributeNames = ["AT", "V", "AP", "RH"]
N, M = X.shape
