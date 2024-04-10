import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from LoadData import *

# Neurons range to explore for the hidden layer
neurons_range = range(1, 5)  # Example range

# Cross-validation setup
outer_cv_k = 10  # Number of folds in the outer loop
inner_cv_k = 10  # Number of folds in the inner loop

# Outer Cross-Validation
outer_cv = KFold(n_splits=outer_cv_k, shuffle=True, random_state=42)

# Initialize storage for the results
optimal_neurons = []
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
    
    # Inner Cross-Validation for finding the optimal number of neurons
    inner_cv = KFold(n_splits=inner_cv_k, shuffle=True, random_state=42)
    neuron_errors = []
    
    for neurons in neurons_range:
        inner_errors = []
        for inner_train_index, inner_test_index in inner_cv.split(X_train_scaled):
            X_inner_train, X_inner_test = X_train_scaled[inner_train_index], X_train_scaled[inner_test_index]
            y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
            
            model = MLPClassifier(hidden_layer_sizes=(neurons,), max_iter=1000, random_state=42,
                                  early_stopping=True, validation_fraction=0.1, 
                        learning_rate_init=0.1,  # Initial learning rate
                        n_iter_no_change=10,  # Number of iterations with no improvement to wait before stopping
                        verbose=False)
            model.fit(X_inner_train, y_inner_train)

            y_inner_pred = model.predict(X_inner_test)
            inner_errors.append(1 - accuracy_score(y_inner_test, y_inner_pred))  # Error rate = 1 - accuracy
            
        neuron_errors.append(np.mean(inner_errors))
    
    # Select the best neuron count for the current outer fold
    optimal_neuron = neurons_range[np.argmin(neuron_errors)]
    optimal_neurons.append(optimal_neuron)
    
    # Retrain on the full training set with the selected neuron count
    model = MLPClassifier(hidden_layer_sizes=(optimal_neuron,), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    test_error = 1 - accuracy_score(y_test, y_test_pred)  # Error rate = 1 - accuracy
    outer_test_errors.append(test_error)
    
    # Print ongoing results
    print(f"Fold {fold}: Optimal Neurons: {optimal_neuron}, Test Error: {test_error}")
    fold += 1

# Final summary
print("\nSummary:")
print("Optimal neuron counts per fold:", optimal_neurons)
print("Test Error per fold:", outer_test_errors)
print("Average Test Error:", np.mean(outer_test_errors))