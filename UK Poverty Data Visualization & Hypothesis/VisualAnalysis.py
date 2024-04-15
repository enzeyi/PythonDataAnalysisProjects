import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel('PovertyInUk.xlsx')

# Data Exploration
print(df.head())  # Check the first few rows to understand the structure of the dataset

# Visualization 1: Trends in Poverty Headcount Ratios Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='year', y='headcount_ratio_international_povline', label='International Poverty Line')
sns.lineplot(data=df, x='year', y='headcount_ratio_lower_mid_income_povline', label='Lower-Mid Income Poverty Line')
sns.lineplot(data=df, x='year', y='headcount_ratio_upper_mid_income_povline', label='Upper-Mid Income Poverty Line')
plt.title('Trends in Poverty Headcount Ratios Over Time')
plt.xlabel('Year')
plt.ylabel('Headcount Ratio')
plt.legend()
plt.show()

# Visualization 2: Income Gap Ratios Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='income_gap_ratio_international_povline', bins=20, kde=True)
plt.title('Income Gap Ratios Distribution')
plt.xlabel('Income Gap Ratio')
plt.ylabel('Frequency')
plt.show()
