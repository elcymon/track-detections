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
    elif mode == 'norm':
        p_u2s_list = np.random.normal(prob_u2s[0],prob_u2s[1],size=detection_data.shape[0])
        p_u2s_list[p_u2s_list > 1] = 1
        p_u2s_list[p_u2s_list < 0] = 0
        
        p_s2s_list = np.random.normal(prob_s2s[0],prob_s2s[1],size=detection_data.shape[0])
        p_s2s_list[p_s2s_list > 1] = 1
        p_s2s_list[p_s2s_list < 0] = 0
    else:
        print('Unknown Mode',mode)
        exit(1)
    for t in range(tsteps):
        print(' ' * 20,end='')
        print('\r',t,end='')
        detection_data.loc[:,t] = 0
        detection_data.loc[:,'visible'] += 1 #always visible
        
        if t == 0:
            if mode == 'norm':
                p_u2s = p_u2s_list
            
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
                p_u2s[unsRows] = p_u2s_list[unsRows]
                
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
                p_s2s[seenRows] = p_s2s_list[seenRows]
                
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

def generate_TPandFN_combinations(s2s,u2s=None):
    if u2s is None:
        u2s = s2s
    TPandFN = {}
    for p_s2s in s2s:
        for p_u2s in u2s:
            name = f'{p_s2s}_{p_u2s}'
            name = name.replace('.','p')
            TPandFN[name] = ((p_s2s,0),(p_u2s,0))
    return TPandFN

if __name__ == '__main__':
    resultsPath = '../data/computational_simulation/'
#    TPandFN = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.8302,0.0395),(0.0539,0.0155)),
#               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.8622,0.0282),(0.1134,0.0270)),
#               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.8323,0.0574),(0.0090,0.0050)), 
#               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.8331,0.0517),(0.0226,0.0126))}
#    
#    TPandFN_1hz = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.7303,0.098),(0.2166 ,0.0684)),
#               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.7710,0.0796),(0.3643,0.1111)),
#               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.5899,0.2607),(0.0623,0.0356)), 
#               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.5954 ,0.2371),(0.1035,0.0613))}
#    
#    TPandFN_4hz = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.7031,0.0608),(0.1200,0.0465)),
#               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.7477,0.0438),(0.2473,0.0812)),
#               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.5898,0.1119),(0.0282,0.0181)), 
#               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.6357,0.0989),(0.0602,0.0397))}
#    
#    TPandFN_40hz =  {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.8206,0.0387),(0.0576,0.0173)),
#               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.8517,0.0289),(0.1227,0.0306)),
#               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.8138,0.0651),(0.0099,0.0054)), 
#               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.8188,0.0548),(0.0249,0.0143))}
    
    TPandFN = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.6784,0.2791),(0.0696,0.0998)),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.7361,0.2577),(0.1696,0.1687)),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.6638,0.2952),(0.0095,0.0263)), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.6656,0.2793),(0.0292,0.0610))}
    
    TPandFN_1hz = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.6499,0.4278),(0.2405 ,0.3476)),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.7121,0.3815),(0.4158,0.4239)),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.5355,0.4549),(0.0670,0.1927)), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.5248 ,0.4356),(0.1149,0.2565))}
    
    TPandFN_4hz = {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.5651,0.3312),(0.1591 ,0.2439)),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.6385,0.3061),(0.3473,0.3496)),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.4545,0.3614),(0.0302,0.0926)), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.4737 ,0.3656),(0.0783,0.1770))}
    
    TPandFN_40hz =  {'mobilenetSSD-10000-th0p5-nms0p0-iSz124':((0.6676,0.2820),(0.0750 ,0.1087)),
               'mobilenetSSD-10000-th0p5-nms0p0-iSz220':((0.7259,0.2578),(0.1841,0.1842)),
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128':((0.6437,0.3026),(0.0105,0.0299)), 
               'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224':((0.6470 ,0.2830),(0.0327,0.0714))}
    mode = 'mean'
    s2s = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999,1.0]
    TPandFN_var = generate_TPandFN_combinations(s2s)
    process_computational_simulation(resultsPath,TPandFN_var,f'v2litters-comp-sim-{mode}-100k','var_s2s_u2s',mode=mode)
#    detection_data = simulate_steps(0.8547,0.0939,100,100)