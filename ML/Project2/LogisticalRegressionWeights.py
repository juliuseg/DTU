# Code written by Julius Gronager, s204427
# Code written in collaboration with Chat-gpt4

from LoadData import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression on the scaled data
optimal_C = 1  # Use the average optimal C obtained from previous cross-validation
model = LogisticRegression(C=optimal_C, max_iter=1000)
model.fit(X_scaled, y_binary)

# After training, print the weights of the logistic regression model and the intercept
print("Weights (coefficients) of the logistic regression model:")
print(model.coef_)
print("Intercept of the logistic regression model:")
print(model.intercept_)