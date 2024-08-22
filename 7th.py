from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')

# Load the iris dataset
iris = datasets.load_iris()

# Print the total number of samples in the dataset and random samples taken as test data
print("Size of full dataset: 150")
print("Size of each 3-test dataset: 50")

X = iris.data
Y = iris.target
X, Y = shuffle(X, Y)

# Build the K-Means Model
model = KMeans(n_clusters=3, random_state=3425)
model.fit(X)
Y_Pred1 = model.labels_
Y_Pred1

cm = confusion_matrix(Y, Y_Pred1)
print("\nK-Means Model:")
print("Confusion Matrix\n", cm)
print("Accuracy score =", accuracy_score(Y, Y_Pred1))

# Build the EM Model
model2 = GaussianMixture(n_components=3, random_state=3425)
model2.fit(X)
Y_Pred2 = model2.predict(X)

cm = confusion_matrix(Y, Y_Pred2)
print("\nEM Model:")
print("Confusion Matrix\n", cm)
print("Accuracy score =", accuracy_score(Y, Y_Pred2))
