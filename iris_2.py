from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split
f_train, f_test , l_train, l_test = train_test_split(features, labels, test_size = .5)

#Classifier Algo
#Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf.fit(f_train,l_train)

predict = clf.predict(f_test)
print ("Prediction by Tree")
print (predict)

from sklearn.metrics import accuracy_score
accu_tree = accuracy_score(l_test, predict)

print ("Accuracy of Tree")
print (accu_tree)
print ("")
#KNearestPadosi
from sklearn.neighbors import KNeighborsClassifier as KNC
clf_2 = KNC()

clf_2.fit(f_train,l_train)

predict_2 = clf_2.predict(f_test)
print ("Prediction by KNC")
print (predict_2)

accu_knc = accuracy_score(l_test, predict_2)
print ("Accuracy of KNC")
print (accu_knc)
