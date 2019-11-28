#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:43:37 2019

@author: elcymon
"""

import pandas as pd
import numpy as np
import os

def simulate_steps(p_visible,p_s2s,p_u2s,p_s_u2s,p_uns_u2s,nlitter,tsteps):
    data2analyse = ['seen2seen','seen2unseen','unseen2seen','unseen2unseen','seen','unseen','visible']
    detection_data = pd.DataFrame(columns=data2analyse,index=['L{}'.format(i) for i in range(nlitter)])
    detection_data.loc[:,:] = 0
    seen_litters = None
    for t in range(tsteps):
        print('\r',t,end='')
        detection_data.loc[:,t] = 0
        detection_data.loc[:,'visible'] += 1 #always visible
        
        if t == 0:
            tCondition = np.random.uniform(size=detection_data.shape[0]) < p_uns_u2s 
            detection_data.loc[tCondition,t] = 1
            detection_data.loc[tCondition,'seen'] += 1
            detection_data.loc[~tCondition,'unseen'] += 1
            seen_litters = detection_data[t] == 1
        else:
            #apply condition to total unsRows
            unsRows = detection_data[t-1] == 0
            #break uns2seen transition data into two categories
            s_u2s_rows = unsRows & seen_litters
            uns_u2s_rows = unsRows & (~seen_litters)
            assert((sum(s_u2s_rows) + sum(uns_u2s_rows)) == sum(unsRows))
            
            s_u2s_random = pd.Series(index = detection_data.index)
            s_u2s_random[:] = np.inf
            s_u2s_random[s_u2s_rows] = np.random.uniform(size=s_u2s_rows.sum())
            detection_data.loc[(s_u2s_random < p_s_u2s) & s_u2s_rows, t] = 1
            detection_data.loc[(s_u2s_random < p_s_u2s) & uns_u2s_rows, ['unseen2seen','seen']] += 1
            detection_data.loc[(s_u2s_random >= p_s_u2s) & uns_u2s_rows, ['unseen2unseen','unseen']] += 1
            
            uns_u2s_random = pd.Series(index = detection_data.index)
            uns_u2s_random[:] = np.inf
            uns_u2s_random[uns_u2s_rows] = np.random.uniform(size=uns_u2s_rows.sum())
            detection_data.loc[(uns_u2s_random < p_uns_u2s) & uns_u2s_rows, t] = 1
            detection_data.loc[(uns_u2s_random < p_uns_u2s) & uns_u2s_rows, ['unseen2seen','seen']] += 1
            detection_data.loc[(uns_u2s_random >= p_uns_u2s) & uns_u2s_rows, ['unseen2unseen','unseen']] += 1
            
            
#            u2sRandom = pd.Series(index = detection_data.index)
#            u2sRandom[:] = np.inf
#            u2sRandom[unsRows] = np.random.uniform(size=unsRows.sum())
#            
#            #update uns2seen transition data
#            
#            
#            detection_data.loc[(u2sRandom < p_u2s) & unsRows, t] = 1
#            detection_data.loc[(u2sRandom < p_u2s) & unsRows, ['unseen2seen','seen']] += 1
#            detection_data.loc[(u2sRandom >= p_u2s) & unsRows, ['unseen2unseen','unseen']] += 1
            
            
            #apply conditions to total seen rows
            seenRows = detection_data[t-1] == 1 # or not of unsRows
            s2sRandom = pd.Series(index = detection_data.index)
            s2sRandom[:] = np.inf
            s2sRandom[seenRows] = np.random.uniform(size=seenRows.sum())
            
            detection_data.loc[(s2sRandom < p_s2s) & seenRows, t] = 1
            detection_data.loc[(s2sRandom < p_s2s) & seenRows, ['seen2seen','seen']] += 1
            detection_data.loc[(s2sRandom >= p_s2s) & seenRows, ['seen2unseen','unseen']] += 1
            
            #update list of litters already seen
            seen_litters = seen_litters | (detection_data[t] == 1)
    print()
    return detection_data
def process_computational_simulation(resultsPath,probDict,experiment,video):
    nlitter = 100
    tsteps = 1000
    for network in probDict.keys():
        print(network)
        csvPath=f'{resultsPath}{experiment}/{experiment}/{video}/{network}/'
        os.makedirs(csvPath,exist_ok=True)
        p_visible,p_s2s,p_u2s,p_s_u2s,p_uns_u2s = probDict[network]
        detection_data = simulate_steps(p_visible,p_s2s,p_u2s,p_s_u2s,p_uns_u2s,nlitter,tsteps)
        detection_data.to_csv(f'{csvPath}{video}-{network}-detection-TPandFN.csv')
    
if __name__ == '__main__':
    resultsPath = '../data/computational_simulation/'
    TPandFN = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':(0.2034,0.8268,0.0437,0.1630,0.0067),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':(0.3880,0.8547, 0.0939,0.2072,0.0197),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':(0.0481,0.8320, 0.0073,0.1015,0.0016), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':(0.1103,0.8319, 0.0186,0.1184,0.0033)}
    process_computational_simulation(resultsPath,TPandFN,'comp-sim-split_u2s-100k','comp-sim')
#    detection_data = simulate_steps(0.8547,0.0939,100,100)