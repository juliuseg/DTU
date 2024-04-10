import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from LoadData import *

# Cross-validation setup
K = 10
CV = KFold(K, shuffle=True)

# Range of neurons to explore in the hidden layer
neurons_range = range(1, 21)  # Exploring 1 to 20 neurons

# Variables to track errors for each model configuration
train_errors = np.empty((K, len(neurons_range)))
test_errors = np.empty((K, len(neurons_range)))

# Cross-validation loop
k = 0
for train_index, test_index in CV.split(X, y):
    print(f'Cross-validation fold {k+1}/{K}...')
    
    # Split into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Standardize data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Loop over each neuron count
    for i, neurons in enumerate(neurons_range):
        # Configure and train the ANN
        model = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=1000, 
                        early_stopping=True, validation_fraction=0.1, 
                        learning_rate_init=0.1,  # Initial learning rate
                        n_iter_no_change=10,  # Number of iterations with no improvement to wait before stopping
                        verbose=False)  # Prints progress messages to stdout for large `max_iter`
        model.fit(X_train_scaled, y_train)
        
        # Predict and calculate mean squared error
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        train_errors[k, i] = mean_squared_error(y_train, y_train_pred)
        test_errors[k, i] = mean_squared_error(y_test, y_test_pred)
    
    k += 1

# Calculate mean errors across folds
mean_train_errors = np.mean(train_errors, axis=0)
mean_test_errors = np.mean(test_errors, axis=0)

# Find optimal number of neurons
optimal_neurons = neurons_range[np.argmin(mean_test_errors)]
print(f'Optimal number of neurons: {optimal_neurons}')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(neurons_range, mean_train_errors, label='Train Error')
plt.plot(neurons_range, mean_test_errors, label='Test Error')
plt.xlabel('Number of Neurons in Hidden Layer')
plt.ylabel('Mean Squared Error')
plt.title('ANN Regression Error vs. Number of Neurons')
plt.legend()
plt.grid(True)
plt.show()