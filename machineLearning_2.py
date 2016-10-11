# Importing all useful libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO

# Loading iris dataset from sklearn.datasets
iris = load_iris()

# Removing some DataRows from target data for testing!
test_idx = [0, 50, 100] # Play around with these values and test!

# Altering the target data!
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Creating testing datasets
testing_target = iris.target[test_idx]
testing_data = iris.data[test_idx]

# creating a classifier for predicting new data values!
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(train_data, train_target)


# Printing the predicted labels
print classifier.predict(testing_data)