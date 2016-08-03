import pickle
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from feature_format.py.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_payments',
                'exercised_stock_options', 'bonus', 'expenses',
                'shared_receipt_with_poi', 'restricted_stock_deferred',
                'total_stock_value', 'long_term_incentive', 
                'from_this_person_to_poi', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Tune n_estimators
print "*Tuning n_estimators"

param_grid = {
              'n_estimators': range(20, 81, 10),          
            }

clf1 = GridSearchCV(GradientBoostingClassifier(min_samples_split=14, min_samples_leaf=1, max_depth=8,  
                                              max_features='sqrt', subsample=0.8, random_state=10 ),
                   param_grid, scoring='roc_auc', iid=False, cv=5)

clf1.fit(features, labels)

new_n_estimators = clf1.best_params_['n_estimators']
print "Best n_estimators is: %d" % (new_n_estimators)
print ""

#Tune max_depth and min_samples_split
print  "*Tuning max_depth and min_samples_split"

param_grid2 = {'max_depth': range(5,16),
              'min_samples_split': range(1, 71)}

clf2 = GridSearchCV(GradientBoostingClassifier(n_estimators=new_n_estimators, min_samples_leaf=1, 
                                              max_features='sqrt', subsample=0.8, random_state=10),
                   param_grid2, scoring='roc_auc', iid=False, cv=5)

clf2.fit(features, labels)

new_max_depth = clf2.best_params_['max_depth']
temp_min_samples_split = clf2.best_params_['min_samples_split']
print "Best max_depth is: %d" % (new_max_depth)
print "Best min_samples_split is: %d" % (temp_min_samples_split)
print ""

#Tune min_samples_split and min_samples_leaf
print "*Tuning min_samples_split and min_samples_leaf"

param_grid3 = {
    'min_samples_split': range(temp_min_samples_split,71),
    'min_samples_leaf': range(1, 9)
}

clf3 = GridSearchCV(GradientBoostingClassifier(n_estimators=new_n_estimators,  max_depth=new_max_depth,  
                                              max_features='sqrt', subsample=0.8, random_state=10),
                   param_grid3, scoring='roc_auc', iid=False, cv=5)

clf3.fit(features, labels)

new_min_split = clf3.best_params_['min_samples_split']
new_min_leaf = clf3.best_params_['min_samples_leaf']
print "Best min_samples_split: %d" % (new_min_split)
print "Best min_samples_leaf: %d" % (new_min_leaf)
print ""

#Create final model
clf = GradientBoostingClassifier(n_estimators=new_n_estimators*10,  max_depth=new_max_depth, 
                                 min_samples_split=new_min_split, min_samples_leaf=new_min_leaf, 
                                 max_features='sqrt', subsample=0.8, random_state=10)

print "Final model: "
print clf
print ""

#Split the data into train/test sets, perform cross validation and output scores
cv = StratifiedShuffleSplit(labels, 1000, random_state=7704)

features_np = np.array(features)
labels_np = np.array(labels)
total_acc = []
total_recall = []
total_precision = []
total_f1 = []

for train_index, test_index in cv:
    features_train_cv, features_test_cv = features_np[train_index], features_np[test_index]
    labels_train_cv, labels_test_cv = labels_np[train_index], labels_np[test_index]
    
    clf = clf.fit(features_train_cv, labels_train_cv)
    pred = clf.predict(features_test_cv)
    
    total_acc.append(accuracy_score(labels_test_cv, pred))
    total_recall.append(recall_score(labels_test_cv, pred))
    total_precision.append(precision_score(labels_test_cv, pred))
    total_f1.append(f1_score(labels_test_cv, pred))
    
print "Average accuracy: %.7f" % (np.mean(total_acc))
print "Average recall: %.7f" % (np.mean(total_recall))
print "Average precision: %.7f" % (np.mean(total_precision))
print "Average f1: %.7f" % (np.mean(total_f1))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)