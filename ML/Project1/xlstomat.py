import pandas as pd
from scipy.io import savemat

# Load your Excel file
data = pd.read_excel('PowerPlant.xls')

# Calculate the size for 1/10th of the dataset
dataset_fraction_size = len(data)

# Now, select only the first 1/10th of the dataset
data_fraction = data.head(dataset_fraction_size)

# Select the first 4 columns as X
X = data_fraction.iloc[:, :4]  # Selecting columns 1 to 4 as predictors

# Select the last column as y
y = data_fraction.iloc[:, -1]  # Selecting the last column as the target variable

# Extract attribute names from the first 4 columns of the DataFrame
attribute_names = list(X.columns)

# Convert the DataFrame to a dictionary with keys 'X', 'y', and 'attributeNames'
data_dict = {'X': X.values, 'y': y.values, 'attributeNames': attribute_names}

# Save the dictionary as a .mat file
savemat('PowerPlant.mat', data_dict)