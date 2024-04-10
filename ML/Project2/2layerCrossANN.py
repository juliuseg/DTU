import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from LoadData import *


# Parameters
neurons_range = range(1, 5)  # Range of neuron counts to explore
outer_cv_k = 10  # Number of folds in the outer loop
inner_cv_k = 10  # Number of folds in the inner loop

# Outer Cross-Validation
outer_cv = KFold(n_splits=outer_cv_k, shuffle=True, random_state=42)

# Initialize lists to store results
optimal_neurons_per_fold = []
test_errors = []

fold_number = 1  # To track the fold number
for train_idx, test_idx in outer_cv.split(X):
    print(f"Outer Fold {fold_number}/{outer_cv_k}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize the data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inner Cross-Validation to find the optimal number of neurons
    inner_cv = KFold(n_splits=inner_cv_k, shuffle=True, random_state=42)
    mean_inner_errors = []

    for neurons in neurons_range:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_scaled):
            X_inner_train, X_inner_val = X_train_scaled[inner_train_idx], X_train_scaled[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]

            model = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=1000, 
                        early_stopping=True, validation_fraction=0.1, 
                        learning_rate_init=0.1,  # Initial learning rate
                        n_iter_no_change=10,  # Number of iterations with no improvement to wait before stopping
                        verbose=False)  # Prints progress messages to stdout for large `max_iter`
            model.fit(X_train_scaled, y_train)
            y_inner_pred = model.predict(X_inner_val)
            inner_errors.append(mean_squared_error(y_inner_val, y_inner_pred))

        mean_inner_errors.append(np.mean(inner_errors))

    # Find the optimal number of neurons
    optimal_neurons = neurons_range[np.argmin(mean_inner_errors)]
    optimal_neurons_per_fold.append(optimal_neurons)

    # Retrain with the optimal neuron count on the entire training set and evaluate on the test set
    model = MLPRegressor(hidden_layer_sizes=(optimal_neurons,), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    test_error = mean_squared_error(y_test, y_test_pred)
    test_errors.append(test_error)

    # Print the results for the current fold
    print(f"  Optimal number of neurons: {optimal_neurons}")
    print(f"  Test MSE: {test_error}\n")
    
    fold_number += 1  # Increment fold number for the next iteration

# Print the final average results
print("Final Results:")
print("Optimal number of neurons per fold:", optimal_neurons_per_fold)
print("Test MSE per fold:", test_errors)
print("Average test MSE:", np.mean(test_errors))
