import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('bdims.csv')

# Select the variables focused on in Task 5
variables = ['bia.di', 'hgt']  

for variable in variables:
    # Extracting independent and dependent variables
    X = data[[variable]]
    y = data['wgt']
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict weight
    y_pred = model.predict(X)
    
    # Residual plot
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y_pred, y=y - y_pred, lowess=True, color="g")
    plt.title(f'Residual Plot for {variable}')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.show()
    
    # Standard error
    se = np.sqrt(mean_squared_error(y, y_pred))
    
    # R-squared
    r2 = r2_score(y, y_pred)
    
    # Print model summary
    print(f"Linear Regression Model Summary for {variable}:")
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
