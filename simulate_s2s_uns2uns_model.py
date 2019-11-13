#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:43:37 2019

@author: elcymon
"""

import pandas as pd
import numpy as np

def simulate_steps(p_s2s,p_u2s,nlitter,tsteps):
    data2analyse = ['seen2seen','seen2unseen','unseen2seen','unseen2unseen','seen','unseen','visible']
    detection_data = pd.DataFrame(columns=data2analyse,index=['L{}'.format(i) for i in range(nlitter)])
    detection_data.loc[:,:] = 0
    for t in range(tsteps):
        print('\r',t,end='')
        detection_data.loc[:,t] = 0
        detection_data.loc[:,'visible'] += 1 #always visible
        
        if t == 0:
            tCondition = np.random.uniform(size=detection_data.shape[0]) < p_u2s
            detection_data.loc[tCondition,t] = 1
            detection_data.loc[tCondition,'seen'] += 1
            detection_data.loc[~tCondition,'unseen'] += 1
        else:
            for l in detection_data.index:
                if detection_data.loc[l,t-1] == 0:
                    if np.random.uniform() < p_u2s:
                        detection_data.loc[l,t] = 1
                        detection_data.loc[l,'unseen2seen'] += 1
                        detection_data.loc[l,'seen'] += 1
                    else:
                        detection_data.loc[l,'unseen2unseen'] += 1
                        detection_data.loc[l,'unseen'] += 1
                        
                elif detection_data.loc[l,t-1] == 1:
                    if np.random.uniform() < p_s2s:
                        detection_data.loc[l,t] = 1
                        detection_data.loc[l,'seen2seen'] += 1
                        detection_data.loc[l,'seen'] += 1
                    else:
                        detection_data.loc[l,'seen2unseen'] += 1
                        detection_data.loc[l,'unseen'] += 1
                else:
                    print('This should not happen')
    
#            unseenMask = detection_data[t-1] == 0
#            unseenCondition = np.random.uniform(size=unseenMask.sum()) < p_u2s
#            detection_data.loc[,t]
    print()
    return detection_data
def process_computational_simulation(resultsPath,probDict):
    nlitter = 100
    tsteps = 1000
    video = 'comp-sim'
    for network in probDict.keys():
        print(network)
        p_s2s,p_u2s = probDict[network]
        detection_data = simulate_steps(p_s2s,p_u2s,nlitter,tsteps)
        detection_data.to_csv(resultsPath + video + '/' + video + '-' + network + '/' + network + '-TPandFN.csv')
    
if __name__ == '__main__':
    resultsPath = '../data/computational_simulation/'
    TPandFN = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':(0.8268,0.0437),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':(0.8547, 0.0939),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':(0.8320, 0.0073), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':(0.8319, 0.0186)}
    process_computational_simulation(resultsPath,TPandFN)
#    detection_data = simulate_steps(0.8547,0.0939,100,100)