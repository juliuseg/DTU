from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from LoadData import *

# Assuming X and y are already defined

# Define the range of lambda values to explore
lambdas = np.power(10.0, range(-5, 9))

# Setup for cross-validation
outer_cv_k = 10  # Number of folds in the outer loop
inner_cv_k = 10  # Number of folds in the inner loop

# Outer Cross-Validation setup
outer_cv = KFold(n_splits=outer_cv_k, shuffle=True, random_state=42)

# Storage for validation errors and test errors
validation_errors = {lambda_val: [] for lambda_val in lambdas}
outer_test_errors = []
generalization_errors = []

# Outer loop
for train_index, test_index in outer_cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Standardize data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inner Cross-Validation for lambda
    inner_cv = KFold(n_splits=inner_cv_k, shuffle=True, random_state=42)
    
    for lambda_val in lambdas:
        lambda_errors = []
        
        for inner_train_index, inner_test_index in inner_cv.split(X_train_scaled):
            X_inner_train, X_inner_test = X_train_scaled[inner_train_index], X_train_scaled[inner_test_index]
            y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
            
            model = Ridge(alpha=lambda_val)
            model.fit(X_inner_train, y_inner_train)
            y_inner_pred = model.predict(X_inner_test)
            lambda_errors.append(mean_squared_error(y_inner_test, y_inner_pred))
        
        validation_errors[lambda_val].append(np.mean(lambda_errors))
    
    # Select the best lambda based on validation error
    optimal_lambda = min(validation_errors, key=lambda k: np.mean(validation_errors[k]))
    generalization_error = np.mean(validation_errors[optimal_lambda])
    generalization_errors.append(generalization_error)
    
    # Retrain on the full training set with the selected lambda and evaluate on the test set
    model = Ridge(alpha=optimal_lambda)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    test_error = mean_squared_error(y_test, y_test_pred)
    outer_test_errors.append(test_error)

# Compute the final estimate of the generalization error
final_generalization_error = np.mean(generalization_errors)

# Print final results
print(f"Final Estimate of Generalization Error: {final_generalization_error}")
print(f"Outer Test Errors: {outer_test_errors}")
print(f"Average Outer Test Error: {np.mean(outer_test_errors)}")