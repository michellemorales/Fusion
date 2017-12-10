#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import re
import glob
import pandas
import numpy as np
import scipy.stats

def process_files(multimodal_files):
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

def aggregate_feature_files(feature_files):
    all_dict = {}
    
    for feature in feature_files:    
        for feature_file in feature_files[feature]:
            p_id = get_participant_id(feature_file)
            if p_id == None:
                print('Ignoring ' + feature_file)
            else:
                if p_id in all_dict:
                    all_dict[p_id].append(feature_file)
                else:
                    all_dict[p_id] = [feature_file]
                    
    return all_dict

def get_feature_files():
    return {
            'audio_features': glob.glob('DAIC_WOZ/Audio_Features/[0-9]*.csv'),
            'ling_features': glob.glob('DAIC_WOZ/Ling_Features/[0-9]*.csv'),
            'video_features': glob.glob('DAIC_WOZ/Video_Features/[0-9]*.csv')
            }
    
def get_participant_id(feature_file):
    #reg ex to match id number in the file path
    pattern = re.compile('.*/(\d+)[^/]+$')
    
    match = pattern.match(feature_file)
    
    if match == None:
        return None
    else:
        return match.group(1)
    