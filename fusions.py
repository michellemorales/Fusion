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


if __name__ == "__main__":
    print('Running fusion experiments...')
    # TODO:
    
    # Find the folder with the data
    # Ask user where it is later

    # Get data for all participants
    all_audio = glob.glob('DAIC_WOZ/Audio_Features/[0-9]*.csv')
 #   print(all_audio)
    
    all_ling = glob.glob('DAIC_WOZ/Ling_Features/[0-9]*.csv')
#    print(all_ling)
    
    all_video = glob.glob('DAIC_WOZ/Video_Features/[0-9]*.csv')
#    print(all_video)
    
    all_labels = ['DAIC_WOZ/Labels/training_split.csv', 
                  'DAIC_WOZ/Labels/dev_split.csv']
    
    all_labels_df = None
    
    for f in all_labels:
        df = pandas.read_csv(f)
        
        if all_labels_df is None:
            all_labels_df = df
        else:
            all_labels_df = all_labels_df.append(df)
            
    all_labels_small_df = all_labels_df[['Participant_ID', 'PHQ_Score']]  #was PHQ_Binary
    all_labels_small_df.to_csv("/Users/tammy/Desktop/all_labels_small_df.csv")
    
    # With the above three variables, create a dictionary, key is the id number
    # value is the count and it should be 3
    all_dict = feature_file_utils.aggregate_feature_files(all_audio, all_ling, all_video)
                
    # Perform early fusion
    all_data = None
    
#    for k in all_dict:
#        if not str(int(k)+1) in all_dict:
#            print(int(k)+1, 'is not an id in the files')
    
    from sklearn.preprocessing import Imputer
    
    for k in all_dict:
        col_names, col_data = early_fusion(all_dict[k])
        
        #if bad data, skip the loop
        if np.isnan(col_data).any() == True:
            print("Skipping " + k)
            continue
            
            # impute here
        
        col_names.append('Participant_ID')
        col_data.append(k)  #append the key(participant_id) that is in all_dict
        
        if all_data == None:
            all_data = dict(zip(col_names, [[x] for x in col_data]))
        else:
            for i in range(len(col_names)):
                all_data[col_names[i]].append(col_data[i])
 
    all_data_df = pandas.DataFrame(all_data)
    all_data_df.to_csv("/Users/tammy/Desktop/all_data_df.csv") #Save it so no need to recreate it

    
    all_data_merged = pandas.merge(all_data_df, all_labels_small_df, on='Participant_ID', how='inner')
    
    labels = all_data_merged[['PHQ_Score']]  #was PHQ_Binary
    data = all_data_merged.drop('PHQ_Score', axis=1)  #was PHQ_Binary
    
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
