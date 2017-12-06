#!/usr/bin/env python

""" This is module for performing different multimodal fusion experiments """
import glob  # this is for specifying pathnames according to a pattern
import pandas
import scipy.stats
import numpy as np
import re
from sklearn import svm
from sklearn.metrics import mean_squared_error
import feature_file_utils


def early_fusion(multimodal_files):
    """
    This function fuses unimodal data files into one multimodal csv file
    :param multimodal_files: a list of unimodal csv files
    :return: mm_names, mm_feats
        WHERE
        list mm_names is the feature names
        list mm_feats is a list of feature lists
    """
    # Statistics we will use to combine features at the frame/sentence level to the participant level
    stats_names = ['max', 'min', 'mean', 'median', 'std', 'var', 'kurt', 'skew', 'percentile25', 'percentile50',
                   'percentile75']
    mm_feats = []
    mm_names = []
    # Process each unimodal data file
    print('Processing unimodal files...')
    for feat_file in multimodal_files:
        df = pandas.read_csv(feat_file, header='infer')
        feature_names = df.columns.values
        for feat in feature_names:
            # Feature vector
            vals = df[feat].values
            # Run statistics
            maximum = np.nanmax(vals)
            minimum = np.nanmin(vals)
            mean = np.nanmean(vals)
            median = np.nanmedian(vals)
            std = np.nanstd(vals)
            var = np.nanvar(vals)
            kurt = scipy.stats.kurtosis(vals)
            skew = scipy.stats.skew(vals)
            percentile25 = np.nanpercentile(vals, 25)
            percentile50 = np.nanpercentile(vals, 50)
            percentile75 = np.nanpercentile(vals, 75)
            names = [feat.strip() + "_" + stat for stat in stats_names]
            feats = [maximum, minimum, mean, median, std, var, kurt, skew, percentile25, percentile50, percentile75]
            for n in names:
                mm_names.append(n)
            for f in feats:
                mm_feats.append(f)
    print('Done combining modalities!')
    return mm_names, mm_feats


def predict_regression(train_data, test_data, train_labels, test_labels):
    """
    This function predicts depression score using an SVM regression algorithm 
    (SVR - http://scikit-learn.org/stable/modules/svm.html#regression)

    :param train_data: list of feature lists for training data
    :param train_labels: list of depression score labels for training data
    :param test_data: list of feature lists for test data
    :param test_labels: list of depression score labels for test data
    :return: RMSE
        WHERE
        float RMSE is root mean square error
    """
    clf = svm.SVR()
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    RMSE = mean_squared_error(predictions, test_labels) ** 0.5
    return RMSE


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
    
def get_labels():
    all_labels = ['DAIC_WOZ/Labels/training_split.csv', 
                  'DAIC_WOZ/Labels/dev_split.csv']
    
    all_labels_df = None
    
    for f in all_labels:
        df = pandas.read_csv(f)
        
        if all_labels_df is None:
            all_labels_df = df
        else:
            all_labels_df = all_labels_df.append(df)
            
    return all_labels_df

def read_labels():
    try:
        return pandas.read_csv("all_labels_df.csv")
    except IOError:
        labels = get_labels()
        labels.to_csv("all_labels_df.csv")
        return labels

def isolate_column(column_to_keep, df):
    return df[['Participant_ID', column_to_keep]]

def read_data():
    try:
        return pandas.read_csv("all_data_df.csv")
    except IOError:
        all_data = get_data()
        all_data.to_csv("all_data_df.csv")
        return all_data
    
def get_data():
    feature_files = feature_file_utils.get_feature_files()
    
    all_dict = feature_file_utils.aggregate_feature_files(feature_files)
                
    # Perform early fusion
    all_data = None
    
    for participant_id in all_dict:
        col_names, col_data = early_fusion(all_dict[participant_id])
        
        #if bad data, skip the loop, will replace this with impute
        if np.isnan(col_data).any():
            print("Skipping " + participant_id)
            continue
        
        col_names.append('Participant_ID')
        col_data.append(participant_id)
        
        if all_data == None:
            all_data = dict(zip(col_names, [[x] for x in col_data]))
        else:
            for i in range(len(col_names)):
                all_data[col_names[i]].append(col_data[i])
 
    return pandas.DataFrame(all_data)

if __name__ == "__main__":
    print('Running fusion experiments...')
    # TODO:
    # Find the folder with the data
    # Ask user where it is later
    
    predictor = 'PHQ_Binary'
    
    all_labels_df = read_labels()
    labels_binary_df = isolate_column(predictor, all_labels_df)
    
    all_data_df = read_data()

    all_data_merged = pandas.merge(all_data_df, labels_binary_df, on='Participant_ID', how='inner')
    
    labels = all_data_merged[[predictor]]
    data = all_data_merged.drop(predictor, axis=1)
    
    from sklearn.model_selection import train_test_split
    
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, random_state=0)
    pass
    
#    from sklearn.neighbors import KNeighborsClassifier
#    
#    clf = KNeighborsClassifier(n_neighbors=3)
#    
#    clf.fit(data_train, labels_train)
#    
#    print("Test set predictions: {}".format(classifier.predict(X_test)))
    
    
 #   import sys
 #   sys.stdout = open('dict_file', 'w')
    
#    sys.stdout = open('example_file', 'w')
#    example = glob.glob('example/300*.csv')
    
#    print(early_fusion(example))
    
    # Experimental Set-up : train on train-split.csv and test of dev_split.csv

    # Get depression labels for each participant from splits

    # Perform regression experiments

    # Perform classification experiments
    print('Done!')
