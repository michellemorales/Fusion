#!/usr/bin/env python

""" This is module for performing different multimodal fusion experiments """
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

import early_fusion
import late_fusion

def predict_class(train_data, test_data, train_labels, test_labels):
    """
    This function predicts depression class (depressed/not depressed) using 
    an SVM classification algorithm 
    (SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC )

    :param train_data: list of feature lists for training data
    :param train_labels: list of depression (binary) class labels for training data
    :param test_data: list of feature lists for test data
    :param test_labels: list of depression (binary) class labels for test data
    :return: accuracy
        WHERE
        float accuracy is the percentage of correct predictions
    """

def classifier_grid_search():
    grid = {
# =============================================================================
#             'kNN_3': lambda: KNeighborsClassifier(n_neighbors=3),
#             'kNN_5': lambda: KNeighborsClassifier(n_neighbors=5),
#             'kNN_10': lambda: KNeighborsClassifier(n_neighbors=10),
#             'kNN_20': lambda: KNeighborsClassifier(n_neighbors=20),
#             'kNN_50': lambda: KNeighborsClassifier(n_neighbors=50),
#                         
#             'DT_None': lambda: tree.DecisionTreeClassifier(),
#             'DT_5': lambda: tree.DecisionTreeClassifier(max_depth=5),
#             'DT_8': lambda: tree.DecisionTreeClassifier(max_depth=8),
#             'DT_10': lambda: tree.DecisionTreeClassifier(max_depth=10),
#             
#             'AdaBoost_10': lambda: AdaBoostClassifier(n_estimators=10),
#             'AdaBoost_100': lambda: AdaBoostClassifier(n_estimators=100),
#             'AdaBoost_300': lambda: AdaBoostClassifier(n_estimators=300)
# =============================================================================
            }

    for c in [0.01, 2, 5, 10, 20]:
        for g in [-10, -5, -1, -1e-1, 0, 1e-1, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-15, 1e-18]:
            for k in ['linear', 'poly', 'sigmoid']:
                grid['SVC_c%s_g%s_k%s' % (c, g, k)] = lambda: svm.SVC(C=c, kernel=k, gamma=g)

    return grid

if __name__ == "__main__":
    print('Running fusion experiments...')

    grid = classifier_grid_search()    
    for clf_desc in grid:
        clf_creator = grid[clf_desc]
    
        print "Early fusion", clf_desc, ":", early_fusion.early_fusion(clf_creator)
        #print "Late fusion with voting", clf_desc, ":", late_fusion.late_fusion_voting(clf_creator)
        #print "Late fusion", clf_desc, ":", late_fusion.late_fusion(clf_creator)

# =============================================================================
#     print("Test set predictions: {}".format(clf.predict(data_test)))
#     # Test set predictions: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1 1 0 1 1 0 0] (old one)
# 
#     print("Test set accuracy: {:.2f}".format(clf.score(data_test, labels_test)))
#     
#     data_test.to_csv("data_test_score.csv")
#     labels_test.to_csv("labels_test_score.csv")
# =============================================================================

    # Test set accuracy: 0.67 (old one)
    
# =============================================================================
#     clf2 = svm.SVR()
#     clf2.fit(data_train, labels_train.values.ravel())
#     predictions = clf2.predict(data_test)
#     
#     print("Test set predictions: {}".format(clf2.predict(data_test)))
#     print("Test set accuracy: {:.2f}".format(clf2.score(data_test, labels_test)))
# =============================================================================
    
#    print(predict_regression(data_train, data_test, labels_train.values.ravel(), labels_test))
    
    # Experimental Set-up : train on train-split.csv and test of dev_split.csv

    # Get depression labels for each participant from splits

    # Perform regression experiments

    # Perform classification experiments
    print('Done!')
