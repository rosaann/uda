#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append("../final_project")
sys.path.append("../tools")
import poi_email_addresses as poi_email
import numpy as np
import matplotlib.pyplot as plt
import feature_format

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#print enron_data
print "no. of enron emails:", len(enron_data)

names = enron_data.keys()
numPoi = 0
for name in names:
    if enron_data[name]["poi"]==1:
        numPoi += 1

print "no. of poi:",  numPoi

poi_email_list = poi_email.poiEmails();
thisEmailAdd = enron_data['COLWELL WESLEY']['email_address']
print "add for search is :", thisEmailAdd
n=0
for p_email in poi_email_list:
    if thisEmailAdd == p_email:
        n += 1

print "n of email from Wesley is :", n

enron_data_formated_poi = feature_format.featureFormat(enron_data, ['poi'])
enron_data_formated_totalPayments =  feature_format.featureFormat(enron_data, ['total_payments' ]);
#print enron_data_formated
plt.xlim(0, len(enron_data_formated))
plt.ylim(0, 5)
plt.xticks()
plt.yticks()
plt.scatter(range(len(enron_data_formated)), enron_data_formated_poi, color ='b', label = 'poi')
plt.scatter(range(len(enron_data_formated)), enron_data_formated_totalPayments, color ='r', label = 'payments')
plt.show()