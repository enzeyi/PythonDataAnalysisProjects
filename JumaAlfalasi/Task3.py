import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('bdims.csv')

# Set seed for reproducibility
seed = 42
random.seed(seed)

# Get the variable names
variable_names = data.columns.tolist()

# Randomly select 4 variables
selected_variables = random.sample(variable_names, 4)
print("Randomly selected variables:", selected_variables)

# Generate histograms and descriptive statistics for each selected variable
for variable in selected_variables:
    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data[variable], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {variable}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    # Descriptive statistics
    description = data[variable].describe()
    print(f"Descriptive statistics for {variable}:")
    print(description)
    print('\n')
