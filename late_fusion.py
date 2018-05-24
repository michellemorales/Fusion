#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import classifiers
import feature_file_utils
import labels

def get_data():
    all_feat_files = feature_file_utils.get_feature_files()
    
    all_data = {}
    
    skipped_participant_ids = []
    for feature in all_feat_files:
        for feature_file in all_feat_files[feature]:
            col_names, col_data = feature_file_utils.process_files([feature_file])
        
            participant_id = feature_file_utils.get_participant_id(feature_file)
            
            if participant_id is None:
                print("Skipping " + feature_file)
                continue
            
            #if bad data, skip the loop, will replace this with impute
            if np.isnan(col_data).any():
                skipped_participant_ids.append(participant_id)
                print("Skipping " + participant_id)
                continue
        
            col_names.append('Participant_ID')
            col_data.append(participant_id)
        
            if feature in all_data:  #if feature not in all_data or if not (feature in all_data)
                for i in range(len(col_names)):
                    all_data[feature][col_names[i]].append(col_data[i])
            else:
                all_data[feature] = dict(zip(col_names, [[x] for x in col_data]))
        
    for feature in all_data:
        all_data[feature] = pandas.DataFrame(all_data[feature])
        all_data[feature] = all_data[feature].loc[~all_data[feature]['Participant_ID'].isin(skipped_participant_ids)]
        
    return all_data

def read_data():
    try:
        return {
                 "audio_features": pandas.read_csv("all_data_audio_df.csv"),
                 "ling_features": pandas.read_csv("all_data_ling_df.csv"),
                 "video_features": pandas.read_csv("all_data_video_df.csv")
               }
    except IOError:
        all_data = get_data()
        for feature in all_data:
            all_data[feature].to_csv("all_data_" + feature.replace("_features", "_df.csv"))
        return all_data

def prepare_data(clf_creator):
    features_data = read_data()
    
    fused_train_data = {}
    fused_test_data = {}
    train_participant_ids = None
    for feature in features_data:
        clf = clf_creator()
        all_data_df, all_labels_df = labels.prepare_dataframes(features_data[feature])
    
        if train_participant_ids is None:
            data_train, data_test, labels_train, labels_test = train_test_split(
                all_data_df, all_labels_df, random_state=0
                )
            train_participant_ids = data_train[['Participant_ID']].values.ravel()
        else:
            data_train = all_data_df.loc[all_data_df['Participant_ID'].isin(train_participant_ids)]
            labels_train = all_labels_df.loc[all_labels_df['Participant_ID'].isin(train_participant_ids)]
            
            data_test = all_data_df.loc[~all_data_df['Participant_ID'].isin(train_participant_ids)]
        
        clf = classifiers.train_classifier(data_train, labels_train[[labels.label_column_name]], clf)
        fused_train_data[feature] = clf.predict(data_train)
        fused_test_data[feature] = clf.predict(data_test)

    final_train_df = pandas.DataFrame(fused_train_data)
    final_train_labels_df = all_labels_df.loc[all_labels_df['Participant_ID'].isin(train_participant_ids)][[labels.label_column_name]]
    final_test_df = pandas.DataFrame(fused_test_data)
    final_labels_test = all_labels_df.loc[~all_labels_df['Participant_ID'].isin(train_participant_ids)]
    
    return final_train_df, final_train_labels_df, final_test_df, final_labels_test
    
def late_fusion(clf_creator):
    final_train_df, final_train_labels_df, final_test_df, final_labels_test = prepare_data(clf_creator)
    
    final_clf = clf_creator()    
    classifiers.train_classifier(final_train_df, final_train_labels_df, final_clf)

    return classifiers.score_classifier(final_test_df, final_labels_test[[labels.label_column_name]], final_clf)    

def late_fusion_voting(clf_creator):
    final_train_df, final_train_labels_df, final_test_df, final_labels_test = prepare_data(clf_creator)

    return accuracy_score(
            final_labels_test[[labels.label_column_name]].values.ravel(),
            ((final_test_df['audio_features'] + final_test_df['video_features'] + final_test_df['ling_features']) >= 2).astype(int).values.ravel()
            )