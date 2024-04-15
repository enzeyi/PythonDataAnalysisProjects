import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('bdims.csv')

# Select the variables focused on in Task 5
variables = ['bia.di', 'hgt']  

# Separate regression models for males and females
for variable in variables:
    for gender in [0, 1]:  # 0 for females, 1 for males
        # Extracting data for the specific gender
        gender_data = data[data['sex'] == gender]
        
        # Extracting independent and dependent variables
        X = gender_data[[variable]]
        y = gender_data['wgt']
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict weight
        y_pred = model.predict(X)
        
        # Standard error
        se = np.sqrt(mean_squared_error(y, y_pred))
        
        # R-squared
        r2 = r2_score(y, y_pred)
        
        # Print model summary
        print(f"Linear Regression Model Summary for {variable} and gender {'male' if gender == 1 else 'female'}:")
        print(f"Intercept: {model.intercept_:.2f}")
        print(f"Slope: {model.coef_[0]:.2f}")
        print(f"Standard Error: {se:.2f}")
        print(f"R-squared: {r2:.2f}")
        print("\n")

        # Interpretation
        if r2 >= 0.7:
            print("The model is a good fit.")
            print(f"The slope represents the change in weight for a one-unit increase in {variable}.")
            print("The y-intercept represents the estimated weight when the independent variable is 0.")
        else:
            print("The model is not a good fit and should be interpreted with caution.")
        print("\n")
