#!/usr/bin/python
import numpy
import sys
import pickle
import os
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import naive_bayes
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import feature_selection
from sklearn import ensemble
from sklearn import grid_search
from sklearn import metrics
emailfiles='emails_by_address'
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
#Uses stratifiedshufflesplit to get classifier score
def getscore(features_list, dataset,clf):
  
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(n_splits=50,random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(features,labels): 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        clf.fit(features_train,labels_train)

        predictions=clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            
    try:
        score=(1.0*true_positives/(true_positives+false_negatives))*(1.0*true_positives/(true_positives+false_positives))
    except:
        return 0
    return score
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
my_dataset = data_dict

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list =['salary','to_messages','deleted_items','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','total_stock_value','expenses','from_poi_to_this_person','from_messages','other','from_this_person_to_poi','long_term_incentive','shared_receipt_with_poi','deferred_income']
 #['poi','salary','to_messages','deleted_items','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','total_stock_value','expenses','from_poi_to_this_person','loan_advances','from_messages','other','from_this_person_to_poi','long_term_incentive','shared_receipt_with_poi','deferred_income']
 # You will need to use more features
max=0
# for i in features_list:
#     for j in features_list:
#         for k in features_list:
#            
# 
#             tclf=svm.SVC()
#             feature=['poi',i,j,k]
#             rec=getscore(feature,my_dataset,tclf)
#             if  rec> max:
#                 feat1,feat2,feat3,clf,max=i,j,k,tclf,rec
# print(clf)
# print("features: "+feat1+" & "+feat2+" & "+feat3)
# print("Score: "+str(max))
selected_features=['poi','expenses','from_this_person_to_poi','other']

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
del my_dataset['TOTAL']
# Creating feature 'deleted_items'

for i in my_dataset:


    my_dataset[i]['deleted_items']=0
    total=0
    for f in os.listdir(emailfiles):
        
        if my_dataset[i]['email_address'] in f:

            fullpath = os.path.join(emailfiles, f)
            set=open(fullpath,'r')
            
            for entry in set:
                if 'deleted_items' in entry:


                    my_dataset[i]['deleted_items']=my_dataset[i]['deleted_items']+1
                total=total+1
    if total != 0:
        my_dataset[i]['deleted_items']=float(my_dataset[i]['deleted_items'])
#Taking the absolute value of numbers
for i in my_dataset:
    for j in my_dataset[i]:
        try:
            my_dataset[i][j]=abs(my_dataset[i][j])
        except:
            pass
            # feats={}


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

#features = feature_selection.SelectKBest(feature_selection.chi2, k=3).fit_transform(features, labels)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.  
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.







### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Parameter Tester
# for a in [2,5,10,20]:
#     for b in [None, 10,5,2,1]:
#         for c in [1,2,5,10,20]:
#             for d in ['gini','entropy']:
#                 for e in [True,False]:
#                     tclf=tree.DecisionTreeClassifier(min_samples_split=a,max_depth=b,min_samples_leaf=c,criterion=d,presort=e)

# Example starting point. Try investigating other evaluation techniques!
clf=tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
max_features=None, max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=2, min_samples_split=2,
min_weight_fraction_leaf=0.0, presort=True, random_state=None,
splitter='best')


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, selected_features)