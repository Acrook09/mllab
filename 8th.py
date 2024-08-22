
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the iris dataset into 75%-train data(112 samples) and 25%-test data(38 samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Print the total number of samples in the dataset and random samples taken as test data
print("Size of full iris-dataset: 150")
print("Size of train dataset: 112")
print("Size of test dataset: 38")

# Build k-Nearest Neighbour model
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
classifier.fit(X_train, y_train)

# Predict the test results and print confusion matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion matrix is as follows:\n', cm)

# Print Accuracy Metrics with correct and wrong predictions
print('\nAccuracy of the classifier is: ', accuracy_score(y_test, y_pred))
print(" \nCorrect predictions: ", accuracy_score(y_test, y_pred))
print("  Wrong predictions: ", (1 - accuracy_score(y_test, y_pred)))
