#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:10:39 2017

@author: tammy
"""

import re

def aggregate_feature_files(audio_files, ling_files, video_files):
    all_dict = {}
    
    #reg ex to match id number in the file path
    pattern = re.compile('.*/(\d+)[^/]+$')
    
    for feature_files in [audio_files, ling_files, video_files]:    
        for feature_file in feature_files:
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