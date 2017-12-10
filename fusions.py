#!/usr/bin/env python

""" This is module for performing different multimodal fusion experiments """
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

import early_fusion

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

if __name__ == "__main__":
    print('Running fusion experiments...')
    
    for clf in [svm.SVR(), KNeighborsClassifier(n_neighbors=3)]:
        score = early_fusion.early_fusion(clf)
    
        print(score)

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
