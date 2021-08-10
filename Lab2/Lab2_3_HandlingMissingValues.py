'''Steps for Handling the missing value

Import Libraries
Load data
Seprate Input and Output attributes
Find the missing values and handle it in either way a. Removing data b. Imputation'''

# Step 1: Import Libraries

import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer 

# Step 2: Load Data
        
datasets = pd.read_csv('Data_for_Missing_Values.csv') 
print("\nData :\n",datasets)
print("\nData statistics\n",datasets.describe())

# Step 3: Seprate Input and Output attributes

# All rows, all columns except last 
X = datasets.iloc[:, :-1].values 
  
# Only last column  
Y = datasets.iloc[:, -1].values 

print("\n\nInput : \n", X) 
print("\n\nOutput: \n", Y)

# Step 4: Find the missing values and handle it in either way

# 4a. Removing the row with all null values

datasets.dropna(how='all',inplace=True)
print("\nNew Data :",datasets)

# 4b. Imputation (Replacing null values with mean value of that attribute)

# All rows, all columns except last 
new_X = datasets.iloc[:, :-1].values 
  
# Only last column  
new_Y = datasets.iloc[:, -1].values 


# Using Imputer function to replace NaN values with mean of that parameter value 
imputer = SimpleImputer(missing_values = np.nan,strategy = "mean")

# Fitting the data, function learns the stats 
imputer = imputer.fit(new_X[:, 1:3]) 
  
# fit_transform() will execute those stats on the input ie. X[:, 1:3] 
new_X[:, 1:3] = imputer.transform(new_X[:, 1:3]) 
  
# filling the missing value with mean 
print("\n\nNew Input with Mean Value for NaN : \n\n", new_X)


