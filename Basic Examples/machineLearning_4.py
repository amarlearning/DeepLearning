# Loading useful python modules
from scipy.spatial import distance
from sklearn import tree , neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# To find the Ecludian distance between two point.
def euc(a,b):
    return distance.euclidean(a,b)

# This classifier is defined to predict labels.
class morningClassifier():
    '''
    This classifier with the help of provided feature 
    tries to predict labels for the testing data.
    '''
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    # method to predict values for the desired input.
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    # Finds the correct label for the given feature.
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

# Loading Iris datasets.
iris = load_iris()

# Storing Features in var "X" && Labels in var "y"
X = iris.data 
y = iris.target

# spliting half of the data for training and other half for testing. [randomly spliting]
X_test , X_train , y_test, y_train = train_test_split(X, y, test_size = 0.5) # train_size = 0.5 means half.

#  Either use this pre-defined classifier
    # classifier = tree.DecisionTreeClassifier()
    # classifier = classifier.fit(X_train, y_train)

# Or this one. there are many you can use any.
    # classifier = neighbors.KNeighborsClassifier()
    # classifier = classifier.fit(X_train, y_train)

# Or we can define our own classifier! :D
classifier = morningClassifier()

# Giving Features as Input to train and test features to predict labels. 
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

# Cheacking how accurate our classifier is.
accuracy = accuracy_score(y_test, predicted)

# Thought the dataset is same. But accuracy may change everytime we  run this program.
#  Because we are spliting the datasets for training & Testing Randomly.
print accuracy