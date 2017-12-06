#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:10:39 2017

@author: tammy
"""

import re
import glob

def aggregate_feature_files(feature_files):
    all_dict = {}
    
    #reg ex to match id number in the file path
    pattern = re.compile('.*/(\d+)[^/]+$')
    
    for feature in feature_files:    
        for feature_file in feature_files[feature]:
            match = pattern.match(feature_file)
            if match == None:
                print('Ignoring ' + feature_file)
            else:
                k = match.group(1)
                if k in all_dict:
                    all_dict[k].append(feature_file)
                else:
                    all_dict[k] = [feature_file]
                    
    return all_dict

def get_feature_files():
    return {
            'audio_features': glob.glob('DAIC_WOZ/Audio_Features/[0-9]*.csv'),
            'ling_features': glob.glob('DAIC_WOZ/Ling_Features/[0-9]*.csv'),
            'video_features': glob.glob('DAIC_WOZ/Video_Features/[0-9]*.csv')
            }