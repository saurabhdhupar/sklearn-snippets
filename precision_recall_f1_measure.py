# As usual, use a train/test split to get a reliable F1 score from two classifiers, and
# save it the scores in the provided dictionaries.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)


clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
precision_cf1 = precision(y_test,y_test_pred)
recall_cf1 = recall(y_test,y_test_pred)
score_1 = f1_score(y_test, clf1.predict(X_test))
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(precision_cf1, recall_cf1)
print "Decision Tree F1 score: {:.2f}".format(score_1)

clf2 = GaussianNB()
clf2.fit(X_train, y_train)
precision_cf2 = precision(y_test,y_test_pred)
recall_cf2 = recall(y_test,y_test_pred)
score_2 = f1_score(y_test, clf2.predict(X_test))
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(precision_cf2, recall_cf2)
print "GaussianNB F1 score: {:.2f}".format(score_2)
