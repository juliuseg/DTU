# Code written in collaboration with Chat-gpt4

import sklearn.linear_model as lm
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
)
from scipy.io import loadmat
from sklearn import model_selection
from dtuimldmtools import rlr_validate

import numpy as np
from sklearn.model_selection import KFold
from LoadData import *

K_outer = 10
outer_cv = KFold(n_splits=K_outer, shuffle=True, random_state=42)

baseline_errors = []

for outer_train_index, outer_test_index in outer_cv.split(X):
    X_outer_train, X_outer_test = X[outer_train_index], X[outer_test_index]
    y_outer_train, y_outer_test = y[outer_train_index], y[outer_test_index]

    # Baseline model: Predict using the mean of the training labels
    y_mean = np.mean(y_outer_train)
    y_baseline_pred = np.full_like(y_outer_test, y_mean)

    # Calculate generalization error for baseline model
    baseline_error = np.square(y_outer_test - y_baseline_pred).mean()
    baseline_errors.append(baseline_error)

    # Print generalization error for each fold
    print("Fold Generalization Error (Baseline): {:.4f}".format(baseline_error))
    print("-" * 40)

# Calculate and print the average generalization error across all folds
average_baseline_error = np.mean(baseline_errors)
print("Average Generalization Error (Baseline): {:.4f}".format(average_baseline_error))
print("Test MSE per fold:", baseline_errors)