import pandas as pd

# Load the dataset
data = pd.read_csv('bdims.csv')

# Set seed for reproducibility
seed = 42
# adding name
name = 'JumaAlfalasi'

# Select a random sample of 100 participants
random_sample = data.sample(n=100, random_state=seed)

# Save the random sample to a CSV file
random_sample.to_csv(f'{name}_random_sample.csv', index=False)
