#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import pydotplus
import matplotlib.pyplot as plt
from IPython.display import Image

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#print "features ", features_train
#########################################################
### your code goes here ###
from sklearn import tree

t0 = time()
print "start decisionTree:", round(time()), "s"
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print "training time:", round(time()-t0, 3), "s"
print acc

#dot_data = tree.export_graphviz(clf, out_file = None)
#graph = pydotplus.graph_from_dot_data(dot_data)
#
#Image(graph.create_png())
#########################################################


