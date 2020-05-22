#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:14:21 2020

@author: fan
"""

#%%
def trim_indices(inds, trim_num, trim_thres = 5):
        l = len(inds)
        to_trim = []
        
        for i in range(min(l-1, trim_num)):
                if inds[i+1] - inds[i] >= trim_thres:
                    to_trim.extend(range(i+1))
            
        for i in range(l-1, max(0, l-trim_num-1), -1):
            if inds[i] - inds[i-1] >= trim_thres:
                    to_trim.extend(range(i,l))
                    
        return([inds[i] for i in range(l) if i not in to_trim])
        
#%%