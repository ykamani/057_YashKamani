# Roll No. : 057

# Impoting libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# reading data into main_data

main_data = pd.read_csv("Dataset2.csv")

# labelEncoder Object

label_encoder = preprocessing.LabelEncoder()

for data in main_data:
  print(f"\n\nHeading :- {data}")
  print(list(main_data[data]))
  main_data[data] = label_encoder.fit_transform(main_data[data])
  print(f"\n\nAfter the tranformation of {data}")
  print(list(main_data[data]))

# now zip all the features of atmosphere

combined_features = tuple(zip(main_data["Outlook"], main_data["Temp"], main_data["Wind"], main_data["Humidity"]))
print("After combined!")

print("Outlook, Temp, Wind, Humidity\n\n")
for pair in combined_features:
    print(pair)

# Train Test Division : 90%-10% & Roll No. = 57
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(combined_features, main_data["Class"], test_size = 0.10, random_state = 57)

# create model
model = MultinomialNB()
model.fit(X_train, Y_train)

# Predict Y from X_text
Y_predicted = model.predict(X_test)

print(Y_predicted)

from sklearn import metrics

print(f"Accuracy is :- {metrics.accuracy_score(Y_test, Y_predicted)}")

# print precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


precision = precision_score(Y_test, Y_predicted)
recall = recall_score(Y_test, Y_predicted)


print(f"precision :- {precision}")
print(f"recall :- {recall}")

# Excersice

# Outlook is ’Rainy’, Temperature is ’Mild’, Humidity =’Normal’, and Wind = ’False’

output = model.predict([[1, 2, 0, 2]])
print(f"final prediction :- {output}")

# Outlook is ’Sunny’, Temeprature is ’Cool’, Humidity =’High’, and Wind = ’True’

output = model.predict([[2, 0, 1, 0]])
print(f"final prediction :- {output}")