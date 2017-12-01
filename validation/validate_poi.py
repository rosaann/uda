#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
from time import time
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
import matplotlib.pyplot as plt

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

for feature, label in zip(features, labels) :
    plt.scatter( feature, label )
    
plt.xlabel("salary")
plt.ylabel("poi")
plt.show()
### it's all yours from here forward!  
from sklearn import tree

n_poi_testReal = 0
for poi in labels_test:
    if poi == 1:
        n_poi_testReal += 1
print "n_poi :", n_poi_testReal, "n_all ", len(labels_test)
t0 = time()
print "start decisionTree:", round(time()), "s"
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

n_poi_testPred = 0
for predPoi, realPoi in zip( pred, labels_test):
    if predPoi == 1 and realPoi == 1 :
        n_poi_testPred += 1
print "n_poi_pred :", n_poi_testPred, "n_all_pred ", len(pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print "training time:", round(time()-t0, 3), "s"
print acc

from sklearn.metrics import precision_score, recall_score
precisionCount = precision_score(labels_test, pred)
recall_scoreCount = recall_score(labels_test, pred)
print "precision " , precisionCount
print "recall " , recall_scoreCount















