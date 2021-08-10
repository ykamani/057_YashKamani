#from google.colab import drive
#drive.mount("/content/drive")

# importing modules
import numpy as np
import pandas as pd

# Creating Two numpy array of size 3 X 2 and 2 X 3
arr1 = np.array([[2,-7],[3, 2],[4, -5]])
print(arr1)
arr2 = np.array([[4, 6, 1],[2, 5, 3]])
print(arr2)

# Randomly Initalizing the array
print(np.random.rand(2, 3))

# Performing matrix multiplication
print(np.dot(arr1, arr2))

# Performing elementwise matrix multiplication
res = [[0 for x in range(len(arr1))] for y in range(len(arr2[0]))]

for i in range(len(arr1)): 
  for j in range(len(arr2[0])): 
    for k in range(len(arr2)): 
      res[i][j] += arr1[i][k] * arr2[k][j]
 
print (res)


# Finding mean of first matrix
mean = np.mean(arr1)
print(mean)

# Converting Numeric entries(columns) of mtcars.csv to Mean Centered Version
main_data = pd.read_csv("mtcars.csv")
del main_data['model']

meancenter = main_data.apply(lambda e: e - e.mean())
print(meancenter.head())

