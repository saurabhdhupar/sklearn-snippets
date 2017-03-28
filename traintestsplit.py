#sklearn-0.17
#How to create a train test split
#organizing imports
#http://scikit-learn.org/0.17/modules/cross_validation.html
from sklearn import datasets
from sklearn import cross_validation

iris = datasets.load_iris()
#Parameters to remember is test_size and random_state
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)


#sklearn-0.18
#How to create a train test split
#organizing imports
#http://scikit-learn.org/0.18/modules/cross_validation.html
from sklearn import datasets
from sklearn import model_selection

iris = datasets.load_iris()
#Parameters to remember is test_size and random_state
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
