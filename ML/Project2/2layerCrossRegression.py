# Code written in collaboration with Chat-gpt4

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from LoadData import *

# Assuming X, y are your data and target variables
# Parameters for lambda
lambdas = np.power(10.0, range(-5, 9))

# Cross-validation setup
outer_cv_k = 10  # Number of folds in the outer loop
inner_cv_k = 10  # Number of folds in the inner loop

# Outer Cross-Validation
outer_cv = KFold(n_splits=outer_cv_k, shuffle=True, random_state=42)

# Initialize storage for the results
optimal_lambdas = []
outer_test_errors = []

# Outer loop
fold = 1
for train_index, test_index in outer_cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Standardize data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inner Cross-Validation for lambda
    inner_cv = KFold(n_splits=inner_cv_k, shuffle=True, random_state=42)
    lambda_errors = []
    
    for lambda_val in lambdas:
        inner_errors = []
        for inner_train_index, inner_test_index in inner_cv.split(X_train_scaled):
            X_inner_train, X_inner_test = X_train_scaled[inner_train_index], X_train_scaled[inner_test_index]
            y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
            
            model = Ridge(alpha=lambda_val)
            model.fit(X_inner_train, y_inner_train)
            y_inner_pred = model.predict(X_inner_test)
            inner_errors.append(mean_squared_error(y_inner_test, y_inner_pred))
            
        lambda_errors.append(np.mean(inner_errors))
    
    # Select the best lambda for the current outer fold
    optimal_lambda = lambdas[np.argmin(lambda_errors)]
    optimal_lambdas.append(optimal_lambda)
    
    # Retrain on the full training set with the selected lambda
    model = Ridge(alpha=optimal_lambda)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    test_error = mean_squared_error(y_test, y_test_pred)
    outer_test_errors.append(test_error)
    
    # Print ongoing results
    print(f"Fold {fold}: Optimal Lambda: {optimal_lambda}, Test Error: {test_error}")
    fold += 1

# Final summary
print("\nSummary:")
print("Optimal lambdas per fold:", optimal_lambdas)
print("Test MSE per fold:", outer_test_errors)
print("Average Test MSE:", np.mean(outer_test_errors))