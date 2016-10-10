import numpy as np
from sklearn.datasets import load_iris as ls
from sklearn import tree

iris = ls()

test_idx = [0,1,2,50,51,52,100,101,102]

#Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis =0)

#testing data

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
