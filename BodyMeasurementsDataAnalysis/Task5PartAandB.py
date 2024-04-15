import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('bdims.csv')

# Select two variables from Task 3 as independent variables
independent_variables = ['bia.di', 'hgt'] 

# Create scatterplots for each variable against weight
for variable in independent_variables:
    plt.figure(figsize=(8, 6))
    plt.scatter(data[variable], data['wgt'], alpha=0.5)
    plt.title(f'Scatterplot of {variable} vs Weight')
    plt.xlabel(variable)
    plt.ylabel('Weight')
    plt.grid(True)
    plt.show()

    # Describe the scatterplot
    correlation = data[variable].corr(data['wgt'])
    if correlation > 0:
        direction = "positive"
    elif correlation < 0:
        direction = "negative"
    else:
        direction = "no"
    
    print(f"Description of the scatterplot of {variable} vs Weight:")
    print(f"Form: The relationship appears to be linear.")
    print(f"Strength: The strength of the relationship is moderate (correlation coefficient = {correlation:.2f}).")
    print(f"Direction: The {direction} direction.")
    print("\n")

 #PART B
 
# Create scatterplots for each variable against weight separately for each gender
for variable in independent_variables:
    plt.figure(figsize=(8, 6))
    # Scatterplot for males
    plt.scatter(data[data['sex'] == 1][variable], data[data['sex'] == 1]['wgt'], color='blue', label='Male', alpha=0.5)
    # Scatterplot for females
    plt.scatter(data[data['sex'] == 0][variable], data[data['sex'] == 0]['wgt'], color='red', label='Female', alpha=0.5)
    plt.title(f'Scatterplot of {variable} vs Weight by Gender')
    plt.xlabel(variable)
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    plt.show()
