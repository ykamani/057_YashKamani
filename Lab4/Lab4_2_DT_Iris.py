# -*- coding: utf-8 -*-
"""2_DT_Iris.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oBynZ0eWrW8Kn5xfawm2eilZgrpYhKBD
"""

#Import scikit-learn dataset library
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

#Load dataset
iris = datasets.load_iris()

# print the names of the 4 features
print("Features :", iris.feature_names, sep="\n")

# print the label type of wine(class_0, class_1, class_2)
print("\nLabels :", iris.target_names)

# print data(feature)shape
print("\nData(feature) shape :", iris.data.shape)

#import the necessary module
from sklearn.model_selection import train_test_split

#split data set into train and test sets
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state = 57)

#Create a Decision Tree Classifier (using Gini)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "gini")

#Train the model using the training sets
dtc.fit(x_train, y_train)

# Predict the classes of test data
y_predicted = dtc.predict(x_test)
print("predicted :",y_predicted)

#print(test_pred.dtype)
from sklearn import metrics
print(y_predicted.dtype)

# Model Accuracy, how often is the classifier correct?
accuracy = metrics.accuracy_score(y_test, y_predicted)
print("Accuracy:",accuracy)

from sklearn.tree import export_graphviz
export_graphviz(dtc,out_file='iris_tree.dot',feature_names=list(iris.feature_names),
                class_names=list(iris.target_names), filled=True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'iris_tree.dot', '-o', 'iris_tree.png', '-Gdpi=600'])

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('iris_tree.png'))
plt.axis('off')
plt.show()

