#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:43:37 2019

@author: elcymon
"""

import pandas as pd
import numpy as np
import os

def simulate_steps(prob_s2s,prob_u2s,mode,nlitter,tsteps):
    data2analyse = ['seen2seen','seen2unseen','unseen2seen','unseen2unseen','seen','unseen','visible']
    detection_data = pd.DataFrame(columns=data2analyse,index=['L{}'.format(i) for i in range(nlitter)])
    detection_data.loc[:,:] = 0
#    seen_litters = None
    if mode == 'mins':
        p_s2s = prob_s2s[0] - prob_s2s[1]
        p_u2s = prob_u2s[0] - prob_u2s[1]
    elif mode == 'maxs':
        p_s2s = prob_s2s[0] + prob_s2s[1]
        p_u2s = prob_u2s[0] + prob_u2s[1]
    elif mode == 'mean':
        p_s2s = prob_s2s[0]
        p_u2s = prob_u2s[0]
    elif mode != 'norm':
        print('Unknown Mode',mode)
        exit(1)
    for t in range(tsteps):
        print('\r',t,end='')
        detection_data.loc[:,t] = 0
        detection_data.loc[:,'visible'] += 1 #always visible
        
        if t == 0:
            if mode == 'norm':
                p_u2s = np.random.normal(prob_u2s[0],prob_u2s[1],size=detection_data.shape[0])
                p_u2s[p_u2s > 1] = 1
                p_u2s[p_u2s < 0] = 0
            
            tCondition = np.random.uniform(size=detection_data.shape[0]) < p_u2s 
            detection_data.loc[tCondition,t] = 1
            detection_data.loc[tCondition,'seen'] += 1
            detection_data.loc[~tCondition,'unseen'] += 1
#            seen_litters = detection_data[t] == 1
        else:
            #apply condition to total unsRows
            unsRows = detection_data[t-1] == 0
#            #break uns2seen transition data into two categories
#            s_u2s_rows = unsRows & seen_litters
#            uns_u2s_rows = unsRows & (~seen_litters)
#            assert((sum(s_u2s_rows) + sum(uns_u2s_rows)) == sum(unsRows))
#            
#            s_u2s_random = pd.Series(index = detection_data.index)
#            s_u2s_random[:] = np.inf
#            s_u2s_random[s_u2s_rows] = np.random.uniform(size=s_u2s_rows.sum())
#            detection_data.loc[(s_u2s_random < p_s_u2s) & s_u2s_rows, t] = 1
#            detection_data.loc[(s_u2s_random < p_s_u2s) & uns_u2s_rows, ['unseen2seen','seen']] += 1
#            detection_data.loc[(s_u2s_random >= p_s_u2s) & uns_u2s_rows, ['unseen2unseen','unseen']] += 1
#            
#            uns_u2s_random = pd.Series(index = detection_data.index)
#            uns_u2s_random[:] = np.inf
#            uns_u2s_random[uns_u2s_rows] = np.random.uniform(size=uns_u2s_rows.sum())
#            detection_data.loc[(uns_u2s_random < p_uns_u2s) & uns_u2s_rows, t] = 1
#            detection_data.loc[(uns_u2s_random < p_uns_u2s) & uns_u2s_rows, ['unseen2seen','seen']] += 1
#            detection_data.loc[(uns_u2s_random >= p_uns_u2s) & uns_u2s_rows, ['unseen2unseen','unseen']] += 1
            
            
            u2sRandom = pd.Series(index = detection_data.index)
            u2sRandom[:] = np.inf
            u2sRandom[unsRows] = np.random.uniform(size=unsRows.sum())
            if mode == 'norm':
                p_u2s = pd.Series(index = detection_data.index)
                p_u2s[:] = np.inf
                p = np.random.normal(prob_u2s[0],prob_u2s[1],size=unsRows.sum())
                p[p > 1] = 1
                p[p < 0] = 0
                p_u2s[unsRows] = p
                
            #update uns2seen transition data
            
            
            detection_data.loc[(u2sRandom < p_u2s) & unsRows, t] = 1
            detection_data.loc[(u2sRandom < p_u2s) & unsRows, ['unseen2seen','seen']] += 1
            detection_data.loc[(u2sRandom >= p_u2s) & unsRows, ['unseen2unseen','unseen']] += 1
            
            
            #apply conditions to total seen rows
            seenRows = detection_data[t-1] == 1 # or not of unsRows
            s2sRandom = pd.Series(index = detection_data.index)
            s2sRandom[:] = np.inf
            s2sRandom[seenRows] = np.random.uniform(size=seenRows.sum())
            
            if mode == 'norm':
                p_s2s = pd.Series(index = detection_data.index)
                p_s2s[:] = np.inf
                p = np.random.normal(prob_s2s[0],prob_s2s[1],size=seenRows.sum())
                p[p > 1] = 1
                p[p < 0] = 0
                p_s2s[seenRows] = p
                
            detection_data.loc[(s2sRandom < p_s2s) & seenRows, t] = 1
            detection_data.loc[(s2sRandom < p_s2s) & seenRows, ['seen2seen','seen']] += 1
            detection_data.loc[(s2sRandom >= p_s2s) & seenRows, ['seen2unseen','unseen']] += 1
            
            #update list of litters already seen
#            seen_litters = seen_litters | (detection_data[t] == 1)
    print()
    return detection_data
def process_computational_simulation(resultsPath,probDict,experiment,video,mode='mean'):
    nlitter = 100
    tsteps = 1000
    for network in probDict.keys():
        print(network)
        csvPath=f'{resultsPath}{experiment}/{experiment}/{video}/{network}/'
        os.makedirs(csvPath,exist_ok=True)
        p_s2s,p_u2s = probDict[network]
        detection_data = simulate_steps(p_s2s,p_u2s,mode,nlitter,tsteps)
        detection_data.to_csv(f'{csvPath}{video}-{network}-detection-TPandFN.csv')
    
if __name__ == '__main__':
    resultsPath = '../data/computational_simulation/'
    TPandFN = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.8302,0.0395),(0.0539,0.0155)),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.8622,0.0282),(0.1134,0.0270)),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.8323,0.0574),(0.0090,0.0050)), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.8331,0.0517),(0.0226,0.0126))}
    
    TPandFN_1hz = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.7303,0.098),(0.2166 ,0.0684)),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.7710,0.0796),(0.3643,0.1111)),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.5899,0.2607),(0.0623,0.0356)), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.5954 ,0.2371),(0.1035,0.0613))}
    
    TPandFN_4hz = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.7031,0.0608),(0.1200,0.0465)),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.7477,0.0438),(0.2473,0.0812)),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.5898,0.1119),(0.0282,0.0181)), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.6357,0.0989),(0.0602,0.0397))}
    
    TPandFN_40hz =  {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.8206,0.0387),(0.0576,0.0173)),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.8517,0.0289),(0.1227,0.0306)),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.8138,0.0651),(0.0099,0.0054)), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.8188,0.0548),(0.0249,0.0143))}
    mode = 'norm'
    process_computational_simulation(resultsPath,TPandFN_1hz,f'comp-sim-ge50-{mode}-1fps-100k','comp-sim',mode=mode)
#    detection_data = simulate_steps(0.8547,0.0939,100,100)