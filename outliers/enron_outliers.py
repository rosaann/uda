#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
sorted(data, reverse=True, key=lambda k: k[1])
#print data
data = np.delete(data, 0,axis = 0)

#print "after", data
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
    
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()




