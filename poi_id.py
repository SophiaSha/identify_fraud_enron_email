#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif #, RFE, chi2
from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm
from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from pprint import pprint

### 1: TARGET DATASET / QUESTION
print "\n 1: TARGET DATASET / QUESTION"
# load the dictionary containing the target dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

# total number of data points?
print "\n How many total number of data points are in the dataset?: ", len(data_dict)

# allocation across classes (POI/non-POI)?
count = 0
for user in data_dict:
    if data_dict[user]['poi'] == True:
        count+=1
print "\n How many POIs are there in the Enron Test-Dataset: ", count

# number of features used?
print "\n Number of features used?: ", len(data_dict.values()[0])

# are there features with many missing values? (Pandas Analyses)
#df = pd.DataFrame.from_records(list(data_dict.values()))
#employees = pd.Series(list(data_dict.keys()))
#print df.dtypes  # Data Types
#df.to_excel('Data_dict.xlsx')
#employees.to_excel('Data_employees.xlsx')
#print df.head()  # Header Information
#head_df = df.head()
#head_df.to_excel('header.xlsx')

#print df.describe(include='all')  # Data
#descibe_df = df.describe(include='all')  # Describing the categorical Series
#descibe_df.to_excel('describe.xlsx')
# feature 'loan_advances' 142 Missing values

# How Many POIs Exist in the POI_names file?
poi_text = 'poi_names.txt'
poi_names = open(poi_text, 'r')
fr = poi_names.readlines()

print "\n How Many Names Exist in poi_names.txt?: ", len(fr[2:])
poi_names.close()


### 2: FEATURES
print "\n 2. FEATURES"
## 2.1: CREATION
target_label = 'poi'
email_features_list = [
    'from_messages',
    'from_poi_to_this_person', 'from_this_person_to_poi',
    'shared_receipt_with_poi', 'to_messages']

financial_features_list = [
    'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses',
    'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred',
    'salary', 'total_payments', 'total_stock_value']

# List of all features
features_list = ['poi'] + financial_features_list + email_features_list


# 2.2.1: OUTLIER REMOVAL

# plot before cleaning the data
for point in data_dict:
    salary = data_dict[point]['salary']
    bonus = data_dict[point]['bonus']

    plt.scatter(salary, bonus, c='red' if data_dict[point]['poi'] else 'green', s=40)
    if point == 'TOTAL':
        plt.annotate('Outlier', xy=(salary, bonus), xytext=(-20, 20), textcoords='offset points',
                     ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()

# 2.2.2: CLEANING
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['LOCKHART EUGENE E'] # no information in the test-Dataset

# Data Wrangling update Names to POI = True
data_dict['BAXTER JOHN C']['poi'] = True
data_dict['DUNCAN JOHN H']['poi'] = True
data_dict['WHITE JR THOMAS E']['poi'] = True
data_dict['BROWN MICHAEL']['poi'] = True
data_dict['DERRICK JR. JAMES V']['poi'] = True
data_dict['FREVERT MARK A']['poi'] = True
data_dict['PAI LOU L']['poi'] = True

# plot again after cleaning
for point in data_dict:
    salary = data_dict[point]['salary']
    bonus = data_dict[point]['bonus']
    plt.scatter(salary, bonus, c='red' if data_dict[point]['poi'] else 'green', s=40)
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()


## 2.3: SELECTION
print "\n 2.3: SELECTION"

def Importance(data_dict, features_list):
    # convert data_dict into readable format
    data = featureFormat(data_dict, features_list)
    target, features = targetFeatureSplit(data)

    clf = DecisionTreeClassifier()
    clf.fit(features, target)

    # calculate the most important features
    importances = clf.feature_importances_ * 100

    pprint([(X, Y) for Y, X in sorted(zip(importances, features_list[1:]), reverse=True)])
    return

# Get features importance before adding new features
print "\nFeatures importance before adding new features:"
Importance(data_dict, features_list)

# Create new feature
def fraction(poi_messages, expenses):
    if poi_messages == 'NaN' or expenses == 'NaN' or poi_messages == 0 or expenses == 0 or poi_messages == 'nan' or expenses == 'nan':
        fraction = 0.
    else:
        fraction = float(poi_messages) / float(expenses)
    return fraction

for name in data_dict:
    data_point = data_dict[name]

    data_point["expenses_from_poi"] = \
        fraction(data_point["from_poi_to_this_person"], data_point["expenses"])
    data_point["expenses_to_poi"] = \
        fraction(data_point["from_this_person_to_poi"], data_point["expenses"])

# Get features importance after adding new features
print "\nFeatures importance after adding new features:"
my_feature_list = features_list + ['expenses_from_poi', 'expenses_to_poi']
Importance(data_dict, my_feature_list)

# Add another feature: 'exercised_stock_options_from_poi’ and ‘exercised_stock_options_to_poi’
#for name in data_dict:
#    data_point = data_dict[name]
#
#    data_point["exercised_stock_options_from_poi"] = \
#        fraction(data_point["from_poi_to_this_person"], data_point["exercised_stock_options"])
#    data_point["exercised_stock_options_to_poi"] = \
#        fraction(data_point["from_this_person_to_poi"], data_point["exercised_stock_options"])
#
## Get features importance after adding second new features
#print "\nFeatures importance after adding new features:"
#my_feature_list = features_list + ['exercised_stock_options_from_poi', 'exercised_stock_options_to_poi']
#Importance(data_dict, my_feature_list)

# store to my_dataset for easy export below
#my_feature_list = features_list
my_dataset = data_dict

# convert my_dataset to numpy.array and split into target & features
data = featureFormat(my_dataset, my_feature_list)
target, features = targetFeatureSplit(data)


# 2.3.1: kBest
num_features = 6#len(features[0])

# f_classif: ANOVA F-value between label/feature for classification tasks.
k_best = SelectKBest(f_classif, k=num_features)
k_best.fit(features, target)
#print "\n k-best: ", k_best.fit_transform(features, target)

# Get a mask, or integer index, of the features selected
#print "\n k-best support:\n ", k_best.get_support(True)
#print "\n k-best params:\n ", k_best.get_params(num_features)
# What is the accuracy / importance score of kBest Selection?
scores = k_best.scores_
unsorted_results = zip(my_feature_list[1:], scores)
#print "\n {0} unsorted SelectkBest features:\n".format(num_features)
#pprint(unsorted_results[:num_features])

sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse= True)
#sorted_results = list(reversed(sorted(unsorted_results, key=lambda x: x[1])))
print "\n {0} best SelectkBest features:\n".format(num_features)
pprint(sorted_results[:num_features])

# print sorted_results
k_best_features = dict(sorted_results[:num_features])
#k_best_features = dict(sorted_results[:num_features])
#print "\n {0} best kBest features: {1}".format(num_features, k_best_features.keys())
#print "Accuracy Score: \n", sorted(scores, reverse=True)
#pprint([(X, Y) for Y, X in sorted(zip(k_best.scores_, my_feature_list[1:]), reverse=True)])
#print "Accuracy Score: \n", k_best.scores_

my_feature_list = ['poi'] + k_best_features.keys()

# convert my_dataset to numpy.array and split into target & features
data = featureFormat(my_dataset, my_feature_list)
#target, features = targetFeatureSplit(data)


## 2.3.3: Recursive
##from sklearn.svm import SVR
##estimator = SVR(kernel="linear")
##Recu = RFE(estimator, 5, step=1)
##Recu.fit(features, target)
#
## What is the accuracy / importance score of Percentile Selection?
##pred_Recu = Recu.predict(features)
#
##Recu_features = dict(pred_Recu[:num_features])
##print "\n{0} best Recursive features: {1}\n".format(num_features, pred_Recu.keys())
#
#
### 2.4: TRANSFORMS
#print "\n 2.4: TRANSFORMS"
## 2.4.1: PCA – Principal Component Analysis
#clf = PCA(n_components=4)
#clf.fit(data)
#print "\n PCA feature components: \n", clf.components_
#print "\n Variance Ratio: \n", clf.explained_variance_ratio_
#print "\n singular Values: \n", clf.singular_values_
#
## plot results
#plt.figure(1, figsize=(4, 3))
#plt.clf()
#plt.axes([.2, .2, .7, .7])
#
#plt.plot(clf.explained_variance_, linewidth=2)
#plt.plot(clf.singular_values_, linewidth=1)
#
#plt.axis('tight')
#plt.xlabel('n_components')
#plt.ylabel('explained_variance_')
#
#plt.show()


## 2.5: SCALING
print "\n 2.5: SCALING"

# data before scaling
#print "\n Unscaled data: \n", data

# 2.5.1: Standard scaler
#data = preprocessing.scale(data, axis=0, with_mean=False, with_std=False, copy=False)
#print "\n with preprocessing Standard scaler: \n", data

# 2.5.2: Minmax scaler
#min_max_scaler = preprocessing.MinMaxScaler()
#data = min_max_scaler.fit_transform(data)
#print "\n with Minmax scaler: \n", data

# 2.5.3: Mean substraction - defect!
#scaler = StandardScaler(copy=True, with_mean=False, with_std=True)
#data = scaler.fit_transform(data)
#print "\n with mean Standard scaler: \n", data

# split my scaled data into target & features
target, features = targetFeatureSplit(data)

### 3: ALGORITHMS
print "\n 3: ALGORITHMS"

# train/test split with train_test_split
#features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=42)

# ALTERNATIVE: StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits = 7 , random_state=42)
#sss.get_n_splits(features, target)

for train_index, test_index in sss.split(features, target):
    features_train, features_test = [features[i] for i in train_index], [features[i] for i in test_index]
    target_train, target_test = [target[i] for i in train_index], [target[i] for i in test_index]


## 3.1: PICK AN ALGORITHM
print "\n 3.1: PICK AN ALGORITHM \n"
# labeled data, yes/ supervised, non-ordered or discrete output

# Decision Tree Classifier
#clf = DecisionTreeClassifier(min_samples_split=7, random_state=4)
#clf = clf.fit(features_train, target_train)
#pred = clf.predict(features_test, target_test)
#print "\n Decision Tree Classifier prediction: \n", pred
#print "\n Decision Tree Classifier importance: \n", sorted(clf.feature_importances_, reverse=True)

# GAUSSIAN NAIVE BASE CLASSIFIER
clf = GaussianNB()
clf = clf.fit(features_train, target_train)
pred = clf.predict(features_test)
print "\n Gaussian Naive Base Classifier: \n"
print " predictions =", pred
print "\n true labels =", target_test
print "\n Gaussian Naive Base Classifier mean accuracy score: \n", clf.score(features_test, target_test)

# Adaboost Classifier
#clf = AdaBoostClassifier(algorithm='SAMME')
#clf = clf.fit(features_train, target_train)
#pred = clf.predict(features_test)
#print "\n AdaBoost prediction: \n", pred
#print "\n AdaBoost importance: \n", sorted(clf.feature_importances_, reverse=True)

## 3.2: Tune your algorithm
print "\n 3.2: Tune your algorithm \n"
#from sklearn.model_selection import GridSearchCV
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svc = svm.SVC()
#clf = GridSearchCV(svc, parameters)
#clf = clf.fit(features_train, target_train)

#params = dict(reduce_dim__n_components=[1, 2, 3], tree__min_samples_split=[2, 4, 6, 8, 10])
#clf = GridSearchCV(clf, param_grid=params, n_jobs=-1, scoring='recall', refit=True)
#clf.fit(features_train, target_train)
#print sorted(clf.cv_results_.keys())

### 4: EVALUATION

## 4.1: VALIDATE

# 4.1.1: train/test split
# cross validation.rain/test split is selected


## 4.2: PICK METRIC
print "\n 4.2: PICK METRIC"
# 4.2.1:	SSE/r2
lin = LinearRegression()
reg = lin.fit(features_train, target_train)
print "\n r-squared score: \n", reg.score(features_test, target_test)

# 4.2.2	Precision
precision = precision_score(pred, target_test, average=None)
print "\n Precision score: \n", precision

# 4.2.3	Recall
recall = recall_score(pred, target_test)
print "\n Recall score: \n", recall

# 4.2.4	F1 Score
F1 = 2 * (precision * recall) / (precision + recall)
print "\n F1 score: \n", F1
    #print f1_score(pred, target_test)

test_classifier(clf, my_dataset, my_feature_list)

# Dump your classifier, dataset, and features_list so
# anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, my_feature_list)
