import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('bdims.csv')

# Set the variables selected in Task 3
selected_variables = ['bia.di', 'che.de', 'wri.gi', 'hgt']

# Generate grouped box plots for each selected variable
for variable in selected_variables:
    plt.figure(figsize=(8, 6))
    data.boxplot(column=variable, by='sex', grid=False)
    plt.title(f'Grouped Boxplot of {variable} by Sex')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Sex (0 = Female, 1 = Male)')
    plt.ylabel(variable)
    plt.show()

    # Describe the data for both genders
    male_data = data[data['sex'] == 1][variable]
    female_data = data[data['sex'] == 0][variable]
    print(f"Five-number summary for {variable}:")
    print("Male:")
    print(male_data.describe())
    print("\nFemale:")
    print(female_data.describe())
    print("\n")
