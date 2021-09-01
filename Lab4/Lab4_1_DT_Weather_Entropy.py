# Import necessary libraries.

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

# Prepare dataset.

#Predictor variables
Outlook = ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast',
            'Rainy', 'Rainy', 'Sunny', 'Rainy','Overcast', 'Overcast', 'Sunny']
Temperature = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
Humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
            'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
Wind = ['False', 'True', 'False', 'False', 'False', 'True', 'True',
            'False', 'False', 'False', 'True', 'True', 'False', 'True']

#Class Label:
Play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# Digitize the data set using encoding

#creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
Outlook_encoded = le.fit_transform(Outlook)
Outlook_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Outllok mapping:",Outlook_name_mapping)

Temperature_encoded = le.fit_transform(Temperature)
Temperature_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Temperature mapping:",Temperature_name_mapping)

Humidity_encoded = le.fit_transform(Humidity)
Humidity_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Humidity mapping:",Humidity_name_mapping)

Wind_encoded = le.fit_transform(Wind)
Wind_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Wind mapping:",Wind_name_mapping)

Play_encoded = le.fit_transform(Play)
Play_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Play mapping:",Play_name_mapping)

print("\n\n")
print("Weather:" ,Outlook_encoded)
print("Temerature:" ,Temperature_encoded)
print("Humidity:" ,Humidity_encoded)
print("Wind:" ,Wind_encoded)
print("Play:" ,Play_encoded)

# Merge different features to prepare dataset

features=tuple(zip(Outlook_encoded,Temperature_encoded,Humidity_encoded,Wind_encoded))
print("Features:",features)

# Train ’Create and Train DecisionTreeClassifier’

#Create a Decision Tree Classifier (using Entropy)
clf_entropy=DecisionTreeClassifier(criterion="entropy")


# Train the model using the training sets
clf_entropy.fit(features,Play_encoded)

# Predict Output for new data

#Predict Output
predicted= clf_entropy.predict([[0,1,0,1]]) # 0:Overcast, 1:Hot, 0:Humidity, 1:Wind 
print("Predicted Value:", predicted)

# Display Decsion Tree 

# from sklearn.tree import export_graphviz
# export_graphviz(clf_entropy,out_file='tree_entropy.dot',
#                feature_names=['outlook','temperature','humidity','wind'],
#                class_names=['play_no','play_yes'], 
#                filled=True)

# # Convert to png
# from subprocess import call
# call(['dot', '-Tpng', 'tree_entropy.dot', '-o', 'tree_entropy.png', '-Gdpi=600'])

# # Display in python
# import matplotlib.pyplot as plt
# plt.figure(figsize = (14, 18))
# plt.imshow(plt.imread('tree_entropy.png'))
# plt.axis('off');
# plt.show();
