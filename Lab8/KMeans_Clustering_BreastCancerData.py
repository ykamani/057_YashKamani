# -*- coding: utf-8 -*-
"""KMeans_Clustering_BreastCancerData.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FzKyPGg2QJ2q36zfxNvQEXSN465gYz05
"""

# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# model
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

from sklearn import datasets

dataset=datasets.load_breast_cancer()
dataset

print(dataset.data.shape)
print(dataset.target.shape)

print(dataset.feature_names)

print(dataset.target_names)

print(dataset.data[0:2])
print(dataset.target[0:2])

print(dataset.data[284:286])
print(dataset.target[284:286])

# 0 for benign and 1 for malignant

plt.scatter(dataset.data[:, 0], dataset.target, c='blue', marker='.')
plt.xlabel('Features')
plt.ylabel('Type of Cancer')
plt.show()

# creating a model
kmeans = KMeans(n_clusters = 7, random_state = 57)

# prediction
prediction = kmeans.fit_predict(dataset.data)
print(prediction)

# shape of clusters

print("\nshapes of clustering\n")
print(kmeans.cluster_centers_.shape)

print("\nclusters\n")
print(kmeans.cluster_centers_)

plt.scatter(dataset.data[:, 0], dataset.target, c='blue', marker='.')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='+')
plt.title('Data points and cluster centroids')
plt.show()

from scipy.stats import mode
labels = np.zeros_like(prediction)
for i in range(7):
  mask = (prediction == i)
  labels[mask] = mode(dataset.target[mask])[0]
  print(labels[mask])
labels

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(dataset.target, labels)

import seaborn as sns

mat = confusion_matrix(dataset.target, labels)
ax = sns.heatmap(mat.T, square=True, annot=True, cbar=False, xticklabels=dataset.target_names, yticklabels=dataset.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label')