# Code written by Julius Gronager, s204427
# Code written in collaboration with Chat-gpt4


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from LoadData import *

# Assuming X, y are your data and target variables, and y has already been converted to binary

# Cross-validation setup
outer_cv_k = 10  # Number of folds in the outer loop

# Outer Cross-Validation
outer_cv = KFold(n_splits=outer_cv_k, shuffle=True, random_state=42)

# Initialize storage for the results
outer_test_errors = []

# Outer loop
fold = 1
for train_index, test_index in outer_cv.split(X):
    y_train, y_test = y_binary[train_index], y_binary[test_index]
    
    # Determine the most frequent class in the training set
    most_frequent = np.bincount(y_train).argmax()
    
    # Predict this class for all test instances
    y_pred = np.full(shape=y_test.shape, fill_value=most_frequent)
    
    # Calculate test error
    test_error = 1 - accuracy_score(y_test, y_pred)  # Error rate = 1 - accuracy
    outer_test_errors.append(test_error)
    
    # Print ongoing results
    print(f"Fold {fold}: Test Error for Most Frequent Class Baseline: {test_error}")
    fold += 1

# Final summary
print("\nSummary:")
print("Test Error per fold for Most Frequent Class Baseline:", outer_test_errors)
print("Average Test Error for Most Frequent Class Baseline:", np.mean(outer_test_errors))
