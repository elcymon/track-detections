# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ntpath
import glob
import cv2 as cv

def get_detection_metric_data(filename):
    return pd.read_csv(filename,index_col=0)

def consecutive_occurrences(row,value):
    value_locations = row == value
    
    consecutives_mask = (value_locations != value_locations.shift()).cumsum()
    
    value_occurrences = value_locations.groupby(consecutives_mask).transform('size') * value_locations
    
    unduplicated_value_occurrences = value_occurrences.loc[value_occurrences.shift() != value_occurrences]
    
    #remove zeros. zeros imply desired value is not the series/row
    result_consecutive_occurrences = unduplicated_value_occurrences[unduplicated_value_occurrences > 0]
    
    return result_consecutive_occurrences
    
def process_consecutive_occurrences(filename):
    metrics_data = get_detection_metric_data(filename)
    frames_data = metrics_data.drop(metrics_data.columns[0:7],axis=1)
    seen2seen = pd.DataFrame(columns = frames_data.columns)
    unseen2unseen = pd.DataFrame(columns = np.arange(1,frames_data.shape[1],1,dtype=np.int))
    
    for row_name,row_data in frames_data.iterrows():
        print(row_name)
        s2s_data = consecutive_occurrences(row_data,1) #seen values are 1
#        print(s2s_data)
#        s2s_data = s2s_data.value_counts() # count frequency of occurence of each detection
#        s2s_data = s2s_data.sort_index()
        seen2seen.loc[row_name,s2s_data.index.values] = s2s_data
    return seen2seen

def integrity_check(videosPath,resultsPath):
    videosDF = pd.DataFrame( columns = ['Raw Data'])
    videosDF.index.name = 'Video'
    videosDF.columns.name = 'numberOfFrames'
    
    for videoFilePath in glob.glob(videosPath + '*.MP4'):
        videoName = ntpath.basename(videoFilePath).replace('.MP4','')
        print(videoName)
        video = cv.VideoCapture(videoFilePath)
        
        videosDF.loc[videoName,'Raw Data'] = video.get(cv.CAP_PROP_FRAME_COUNT)
        
        #loop through results folders to check videos
        for networkFolder in glob.glob(resultsPath + videoName + '/*'):
            networkName = ntpath.basename(networkFolder)
            networkVideo = networkFolder + '/' + videoName + '-' + networkName
            if '608' in networkName:
                networkVideo = networkVideo + '-GT-pruned.avi'
            else:
               networkVideo = networkVideo + '-detection.avi'
            print(networkName)
            video = cv.VideoCapture(networkVideo)
            videosDF.loc[videoName,networkName] = video.get(cv.CAP_PROP_FRAME_COUNT)
    
    videosDF = videosDF.astype(int)
    return videosDF
def summarise_csv_data(resultsPath,resultCategory):
    
    summaryDF = pd.DataFrame()
    summaryCols = ['seen2seen','seen2unseen','P_s2s',\
                'unseen2seen','unseen2unseen','P_s2u',\
                'seen','unseen','visible','P_visible']
    for videoFolder in glob.glob(resultsPath + '*'):
        videoName = ntpath.basename(videoFolder)
        print(videoName,videoFolder)
        for networkFolder in glob.glob(videoFolder + '/*'):
            networkName = ntpath.basename(networkFolder)
            print(networkName)
            csvFile = networkFolder + '/' + videoName + '-' + networkName
            if '608' in networkName:
                #get information about number of unique litters from GT
                #csv file for both pruned and unpruned
                csvDF = pd.read_csv(csvFile + '-GT-unpruned.csv',header=[0,1],index_col=0,
                                    low_memory=False,nrows=10)
                litterIDs = csvDF.columns.get_level_values(0).unique()
                summaryDF.loc[videoName,'nlitter_unpruned'] = len(litterIDs)
                
                csvDF = pd.read_csv(csvFile + '-GT-pruned.csv',header=[0,1],index_col=0,
                                    low_memory=False,nrows=10)
                litterIDs = csvDF.columns.get_level_values(0).unique()
                summaryDF.loc[videoName,'nlitter_pruned'] = len(litterIDs)
                continue
            csvFile = csvFile + '-detection-' + resultCategory + '.csv'
            csvDF = pd.read_csv(csvFile,index_col=0,header=0).loc[:,'seen2seen':'visible']
            
            dataHeader = pd.MultiIndex.from_product([[networkName],summaryCols],names=['Network','Data'])
            
            if len(summaryDF.columns) == 0:
                summaryDF = pd.DataFrame(columns=dataHeader)
                # self.trackerDF.columns.names
            else:
                # print(self.trackerDF.columns.union(dataHeader))
                summaryDF = summaryDF.reindex(columns=summaryDF.columns.union(dataHeader))
#            csvSummary = pd.DataFrame(csvDF.sum()).T
#            print(csvSummary)
#            csvSummary.index = [videoName]
#            print(summaryDF)
#            summaryDF[networkName].update(csvSummary)#,how='left',left_on=[videoName,dataHeader])
#            print(summaryDF)
#            return summaryDF
            csvDFsummary = csvDF.sum()
            csvDFsummary['P_s2s'] = \
                csvDFsummary['seen2seen'] / (csvDFsummary['seen2seen'] + csvDFsummary['seen2unseen'])
            csvDFsummary['P_u2s'] = \
                csvDFsummary['unseen2seen'] / (csvDFsummary['unseen2seen'] + csvDFsummary['unseen2unseen'])
            csvDFsummary['P_visible'] = \
                csvDFsummary['seen'] / (csvDFsummary['visible'])
            summaryDF.loc[videoName,dataHeader] = csvDFsummary.values
#            summaryDF.merge(csvDF.sum(),how='left',on=videoName,left_on=networkName)
#            csvDF = csvDF
#            return csvDF
    return summaryDF

if __name__ == '__main__':
    resultPath = '../data/model_data/'
    videosPath = '../data/mp4/'
    summary_analysis = '../data/summary_analysis/'
    
    TPandFN = summarise_csv_data(resultPath,resultCategory = 'TPandFN')
    
    TPandFN.to_csv(summary_analysis + 'TPandFN.csv')
    