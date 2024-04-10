# Code written in collaboration with Chat-gpt4


import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from LoadData import *

# Lambda values (converted to C for LogisticRegression)
Cs = np.power(10.0, range(-5, 9))  # Cs = 1/lambda

# Cross-validation setup
outer_cv_k = 10  # Number of folds in the outer loop
inner_cv_k = 10  # Number of folds in the inner loop

# Outer Cross-Validation
outer_cv = KFold(n_splits=outer_cv_k, shuffle=True, random_state=42)

# Initialize storage for the results
optimal_Cs = []
outer_test_errors = []

# Outer loop
fold = 1
for train_index, test_index in outer_cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_binary[train_index], y_binary[test_index]
    
    # Standardize data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inner Cross-Validation for C (inverse of lambda)
    inner_cv = KFold(n_splits=inner_cv_k, shuffle=True, random_state=42)
    C_errors = []
    
    for C in Cs:
        inner_errors = []
        for inner_train_index, inner_test_index in inner_cv.split(X_train_scaled):
            X_inner_train, X_inner_test = X_train_scaled[inner_train_index], X_train_scaled[inner_test_index]
            y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
            
            model = LogisticRegression(C=C, max_iter=1000)
            model.fit(X_inner_train, y_inner_train)
            y_inner_pred = model.predict(X_inner_test)
            inner_errors.append(mean_squared_error(y_inner_test, y_inner_pred))
            
        C_errors.append(np.mean(inner_errors))
    
    # Select the best C for the current outer fold
    optimal_C = Cs[np.argmin(C_errors)]
    optimal_Cs.append(optimal_C)
    
    # Retrain on the full training set with the selected C
    model = LogisticRegression(C=optimal_C, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    test_error = mean_squared_error(y_test, y_test_pred)
    outer_test_errors.append(test_error)
    
    # Print ongoing results
    print(f"Fold {fold}: Optimal C: {optimal_C}, Test Error: {test_error}")
    fold += 1

# Final summary
print("\nSummary:")
print("Optimal C values per fold:", optimal_Cs)
print("Test MSE per fold:", outer_test_errors)
print("Average Test MSE:", np.mean(outer_test_errors))