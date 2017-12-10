#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas

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

def prepare_dataframes(all_data_df, column_to_keep):
    all_labels_df = read_labels()
    
    labels_to_keep_df = isolate_column(column_to_keep, all_labels_df)
    
    all_data_merged = pandas.merge(all_data_df, labels_to_keep_df, on='Participant_ID', how='inner')
    
    labels = all_data_merged[[column_to_keep]]
    data = all_data_merged.drop(column_to_keep, axis=1)
    
    return data, labels