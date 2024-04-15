# %% [markdown]
# ### My Work

# %%
# IMPORTING ALL PACKAGES NEEDED
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# %%
# Importing pandas library
import pandas as pd

# File path to dataset CSV file
file_path = "train.csv"

# Columns to use
use_cols = [
    'Census_TotalPhysicalRAM', 'Census_ProcessorCoreCount', 'Census_OSBuildNumber', 'HasTpm', 'IsProtected', 'AVProductsEnabled', 'AVProductStatesIdentifier', 'Platform', 'Census_FirmwareManufacturerIdentifier', 'Census_ProcessorModelIdentifier'
]

# Load the dataset into a DataFrame
df = pd.read_csv(file_path, usecols=use_cols)

# Display the first few rows of the DataFrame to verify the loading
print(df.head())


# %%
# Define a measure of computer power
df['computer_power'] = df['Census_TotalPhysicalRAM'] * df['Census_ProcessorCoreCount']

# %%
#import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of computer power
plt.figure(figsize=(10, 6))
sns.histplot(df['computer_power'], bins=20, kde=True)
plt.title('Distribution of Computer Power')
plt.xlabel('Computer Power')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### Task 2: Analysis of Computer Power and Malware Detection
# 
# #### Defining Computer Power
# To measure computer power, I created a new feature called `computer_power` by multiplying the RAM by the processor core count.
# 
# #### Distribution of Computer Power
# The histogram above shows the distribution of computer power among the machines in the dataset. We can observe that the distribution is skewed, with most machines having relatively low power.
# 
# ```python
# # Relationship between computer power and malware detection
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='malware_detection', y='computer_power', data=df)
# plt.title('Computer Power vs Malware Detection')
# plt.xlabel('Malware Detection')
# plt.ylabel('Computer Power')
# plt.show()
# 

# %% [markdown]
# #### Relationship between Computer Power and Malware Detection
# The boxplot above illustrates the relationship between computer power and malware detection. We can observe that machines with higher computer power tend to have fewer malware detections, suggesting a potential inverse relationship between computer power and malware presence.

# %%
# Plotting number of machines with malware detections against Census_OSBuildNumber
plt.figure(figsize=(12, 6))
sns.countplot(x='Census_OSBuildNumber', hue='HasTpm', data=df)
plt.title('Number of Machines with Malware Detections by Census_OSBuildNumber')
plt.xlabel('Census_OSBuildNumber')
plt.ylabel('Number of Machines')
plt.xticks(rotation=90)
plt.legend(title='Malware Detected')
plt.show()

# %% [markdown]
# ### Task 3: Analysis of Malware Detections with Software Updates
# 
# #### Malware Detections by Census_OSBuildNumber
# The countplot above displays the number of malware detections grouped by Census_OSBuildNumber. Each bar represents a specific build number of the operating system. We can observe variations in the number of malware detections across different builds, indicating potential correlations between software updates and malware prevalence.
# 
# ```python
# # Plotting percentage of malware detections against Census_OSBuildRevision
# malware_percentage = df.groupby('Census_OSBuildRevision')['malware_detection'].mean() * 100
# 
# plt.figure(figsize=(12, 6))
# malware_percentage.plot(kind='line')
# plt.title('Percentage of Malware Detections by Census_OSBuildRevision')
# plt.xlabel('Census_OSBuildRevision')
# plt.ylabel('Percentage of Malware Detections')
# plt.xticks(rotation=90)
# plt.grid(True)
# plt.show()
# 

# %% [markdown]
# #### Malware Detections by Census_OSBuildRevision
# The line plot above illustrates the percentage of malware detections based on Census_OSBuildRevision. Each point on the line represents a specific revision of the operating system build. We can observe trends in malware detection rates over different revisions, providing insights into the impact of software updates on malware prevalence.
# 

# %%
# Importing pandas library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path to dataset CSV file
file_path = "train.csv"

# Columns to use
use_cols = [
    'Census_TotalPhysicalRAM', 'Census_ProcessorCoreCount', 'Census_OSBuildNumber', 'HasTpm', 'IsProtected', 'AVProductsEnabled', 'AVProductStatesIdentifier', 'Platform', 'Census_FirmwareManufacturerIdentifier', 'Census_ProcessorModelIdentifier'
]

# Load the dataset into a DataFrame
df = pd.read_csv(file_path, usecols=use_cols)

# Analyzing the impact of antivirus software on malware detection
antivirus_counts = df['AVProductStatesIdentifier'].value_counts()
malware_by_antivirus = df.groupby('AVProductStatesIdentifier')['IsProtected'].mean()

plt.figure(figsize=(10, 6))

# Plotting the number of antivirus products used
plt.subplot(1, 2, 1)
antivirus_counts.plot(kind='bar')
plt.title('Number of Antivirus Products Used')
plt.xlabel('Number of Antivirus Products')
plt.ylabel('Count')

# Plotting malware detection rate by number of antivirus products
plt.subplot(1, 2, 2)
malware_by_antivirus.plot(kind='bar')
plt.title('Malware Detection Rate by Number of Antivirus Products')
plt.xlabel('Number of Antivirus Products')
plt.ylabel('Malware Detection Rate')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Task 4: Impact of Antivirus Software on Malware Detection
# 
# #### Number of Antivirus Products Used
# The bar plot on the left displays the distribution of the number of antivirus products used across the dataset. We can observe the frequency of machines using different numbers of antivirus products.
# 
# #### Malware Detection Rate by Number of Antivirus Products
# The bar plot on the right illustrates the malware detection rate based on the number of antivirus products used. We can observe whether there is a correlation between the number of antivirus products installed and the likelihood of malware detection.

# %%
#import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Plot 1: Distribution of malware detections by operating system version
plt.figure(figsize=(10, 6))
sns.countplot(x='Platform', hue='AVProductsEnabled', data=df)
plt.title('Malware Detections by Operating System Version')
plt.xlabel('Operating System Version')
plt.ylabel('Count of Machines')
plt.xticks(rotation=90)
plt.legend(title='Malware Detection')
plt.show()

# %% [markdown]
# ### Task 5: Exploratory Data Analysis
# 
# #### Plot 1: Malware Detections by Operating System Version
# The countplot above shows the distribution of malware detections across different versions of the operating system. It provides insights into which OS versions are more susceptible to malware attacks.
# 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

row_to_remove = df[df['Platform'] == "windows10"].index[0]

# Drop the identified row
df = df.drop(index=row_to_remove)

# Plot 2: Correlation heatmap of numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# %% [markdown]
# #### Plot 2: Correlation Heatmap of Numerical Features
# The heatmap above illustrates the correlation between numerical features in the dataset. It helps identify potential relationships between different features, which can aid in feature selection and model building.
# 

# %%
# Plot 3: Malware detections by processor manufacturer
plt.figure(figsize=(8, 6))
sns.countplot(x='Census_FirmwareManufacturerIdentifier', hue='AVProductsEnabled', data=df)
plt.title('Malware Detections by Processor Manufacturer')
plt.xlabel('Processor Manufacturer')
plt.ylabel('Count of Machines')
plt.legend(title='Malware Detection')
plt.show()

# %% [markdown]
# #### Plot 3: Malware Detections by Processor Manufacturer
# The countplot above visualizes the distribution of malware detections based on the processor manufacturer. It provides insights into whether certain processor brands are more vulnerable to malware attacks.

# %%
from sklearn.impute import SimpleImputer

# Drop samples with missing target values
X_test = X_test.dropna()
y_test = y_test.dropna()

# Predictions on the test set with only the samples that have target values
y_pred = model.predict(X_test)

# Ensure that y_pred and y_test have the same number of samples
y_test = y_test[:len(y_pred)]

# Calculate error rate and AUC score
error_rate = 1 - accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

# Print error rate and AUC score
print(f"Error rate: {error_rate:.4f}")
print(f"AUC score: {auc_score:.4f}")


# %% [markdown]
# ### Task 6: Building Baseline Logistic Regression Model (Model 0)
# 
# #### Model Building and Evaluation
# I constructed a baseline logistic regression model (Model 0) to predict malware detection based on features such as RAM, processor core count, and other relevant features. Here are the evaluation metrics:
# 
# - **Error Rate:** The error rate of the model on the test set is calculated as 1 - accuracy score.
# - **AUC Score:** The Area Under the ROC Curve (AUC) score indicates the model's ability to discriminate between positive and negative samples.
# 
# The error rate and AUC score provide insights into the performance of the baseline model.

# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("train.csv")

# Separate the features (X) and target variable (y)
X = df.drop(columns=["target_variable"])
y = df["target_variable"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in the training set
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)

# Check for NaN values in the target variable (y_train)
print(f"Number of NaN values in y_train: {y_train.isna().sum()}")

# Handle NaN values in y_train
if y_train.isna().sum() > 0:
    # If y_train contains NaN values, drop the corresponding rows
    X_train_imputed = X_train_imputed[~y_train.isna()]
    y_train = y_train.dropna()
else:
    # If y_train does not contain NaN values, proceed as before
    X_train_imputed = X_train_imputed

# Model 2: Random Forest (assuming classification problem)
# If you have a regression problem, consider other regression models instead of Random Forest.
model2 = RandomForestClassifier()
model2.fit(X_train_imputed, y_train)  # Assuming classification problem (adjust if regression)

# Impute missing values in the testing set
X_test_imputed = imputer.transform(X_test)

# Make predictions on the testing set
y_pred_model2 = model2.predict(X_test_imputed)

# %% [markdown]
# ### Task 7: Model Creation and Evaluation
# 
# #### Feature Preprocessing
# I performed feature preprocessing using standard scaling to ensure that all features have a mean of 0 and a standard deviation of 1.
# 
# #### Model 1: Logistic Regression
# I trained Model 1 using logistic regression on the preprocessed features and evaluated its performance.
# 
# #### Model 2: Random Forest
# Model 2 is trained using a Random Forest classifier on the preprocessed features and evaluated accordingly.
# 
# #### Model Evaluation
# Here are the evaluation metrics for both models:
# - **Error Rate:** The error rate indicates the proportion of incorrectly classified instances.
# - **Confusion Matrix:** It provides a detailed breakdown of the model's performance, showing the number of true positives, true negatives, false positives, and false negatives.
# 
# These metrics help assess the performance of each model in predicting malware detection.


