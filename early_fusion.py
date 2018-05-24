#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 18:23:22 2017

@author: tammy
"""
import pandas
import numpy as np
from sklearn.model_selection import train_test_split

import feature_file_utils
import labels
import classifiers

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
                
    all_data = None
    
    for participant_id in all_dict:
        col_names, col_data = feature_file_utils.process_files(all_dict[participant_id])
        
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

def early_fusion(clf_creator):
    all_data_df, all_labels_df = labels.prepare_dataframes(read_data())
    
    all_labels_df = all_labels_df[[labels.label_column_name]]
    
    data_train, data_test, labels_train, labels_test = train_test_split(
            all_data_df, all_labels_df, random_state=0
    )

    clf = clf_creator()
    classifiers.train_classifier(data_train, labels_train, clf)
    
    return classifiers.score_classifier(data_test, labels_test, clf)