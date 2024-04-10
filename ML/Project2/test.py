import numpy as np

data = np.array([
    [33, 4, 0], # y = 1
    [28, 2, 1], # y = 2
    [30, 3, 0], # y = 3
    [29, 5, 0]  # y = 4
])

def calculate_classification_error_for_split(split_data):
    total_count = np.sum(split_data)
    if total_count == 0:  # To handle the case where a split might have no data
        return 0
    max_count = np.max(split_data)
    error = (max_count / total_count)
    return error

# Calculate the error for each value of x7
error_x7_0 = calculate_classification_error_for_split(data[:, 0])
error_x7_1 = calculate_classification_error_for_split(data[:, 1])
error_x7_2 = calculate_classification_error_for_split(data[:, 2])

error_x7_0, error_x7_1, error_x7_2
print(error_x7_0, error_x7_1, error_x7_2)