import pandas as pd

msg = pd.read_csv('naivetext.csv', names=['message', 'label'])

print('Total instances in the dataset:', msg.shape[0])

msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

X = msg.message
Y = msg.labelnum

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

print('\nDataset is split into Actual Training and Testing samples')
print('Total training instances :', xtrain.shape[0])
print('Total testing instances :', xtest.shape[0])

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

xtrain_dtm = count_vect.fit_transform(xtrain)  # Sparse matrix
xtest_dtm = count_vect.transform(xtest)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(xtrain_dtm, ytrain)

predicted = clf.predict(xtest_dtm)

print('\nPredicted instances are:')

for doc, p in zip(xtest, predicted):
    pred = 'pos' if p == 1 else 'neg'
    print("%s -> %s" % (doc, pred))

from sklearn import metrics

print('\n-------Accuracy metrics---------')
print('Confusion matrix \n', metrics.confusion_matrix(ytest, predicted))
print('\nAccuracy of the classifier is', metrics.accuracy_score(ytest, predicted))
print('Precision :', metrics.precision_score(ytest, predicted))
print('Recall :', metrics.recall_score(ytest, predicted))
