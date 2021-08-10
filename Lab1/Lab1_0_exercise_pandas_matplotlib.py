# importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from google.colab import drive
#drive.mount('/content/drive')

# reading data from csv
main_data = pd.read_csv('Data_for_Transformation.csv')

# 1st plot
print(main_data.head())
plt.scatter(main_data["Age"], main_data["Salary"])
plt.show()

# 2nd plot
plt.hist(main_data["Salary"], bins = 10, color = "blue")
plt.show()

# 3rd plot
fig_size = plt.figure(figsize=(7, 5))
plt.bar(main_data["Country"], main_data["Salary"], color="green")
plt.xlabel("Salary")
plt.ylabel("Country")
plt.title("Bar chart of Country vs Salary")
plt.show()
