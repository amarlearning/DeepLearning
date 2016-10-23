'''
In this Machine Learning Program, We have used a pre-defined classifier
named 'DecisionTreeClassifier' defined in sklearn
'''
from sklearn import tree

# In Features, '0' means Bumpy && '1' means Smooth!
features = [[130, 1], [140, 1], [150, 0], [170, 0]]

# In Labels, '0' denotes Apple && '1' denote Orange!
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

# Input Array whoes value is to be predicted!
# You can play around with this.
Input_array = [[150, 0], [120, 1]]

# This will return an array of predicted values!
predicted_values = classifier.predict(Input_array)

# Printing out all the Predicted values
for predicted_value in predicted_values:
    if(predicted_value == 1):
        print "Orange"
    else:
        print "Apple"