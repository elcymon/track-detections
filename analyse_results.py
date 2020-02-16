# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ntpath
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import os
from collections import Counter
from matplotlib.ticker import MaxNLocator
def get_detection_metric_data(filename,visible_threshold=1):
    
    rawDataDF = pd.read_csv(filename,index_col = 0, header = 0)
    rawDataDF = rawDataDF.loc[rawDataDF['visible'] >= visible_threshold,:]#drop rows that are below threshold
    
    return rawDataDF
    

def consecutive_occurrences(row,value):
    value_locations = row == value
    
    consecutives_mask = (value_locations != value_locations.shift()).cumsum()
    
    value_occurrences = value_locations.groupby(consecutives_mask).transform('size') * value_locations
    
    unduplicated_value_occurrences = value_occurrences.loc[value_occurrences.shift() != value_occurrences]
    
    #remove zeros. zeros imply desired value is not the series/row
    result_consecutive_occurrences = unduplicated_value_occurrences[unduplicated_value_occurrences > 0]
    
    return result_consecutive_occurrences
def populate_duration_frequency(df,data,networkName):
    for d in data:
        if d in df.index:
#            print(df.loc[d,(networkName,'Frequency')],np.isnan(df.loc[d,(networkName,'Frequency')]))
            if np.isnan(df.loc[d,(networkName,'Frequency')]):
#                print('assign 0')
                df.loc[d,(networkName,'Frequency')] = 0
            df.loc[d,(networkName,'Frequency')] += 1
        else:
            df.loc[d,(networkName,'Frequency')] = 1
    return df

def update_duration_df_header(df,dataHeader):
    
    if len(df.columns) == 0:
        df = pd.DataFrame(columns = dataHeader)
    else:
        df = df.reindex(columns=df.columns.union(dataHeader))
    return df

def process_consecutive_occurrences(resultsPath,resultCategory,visible_threshold=1):
#    seen2seen = pd.DataFrame(columns = frames_data.columns)
    cols = ['Frequency','PctFreq','FreqxSteps','PctFreqxSteps']
    s2s_duration = pd.DataFrame()#columns=['Frequency'])
    s2s_duration.index.name = 'Steps'
    uns2uns_duration = pd.DataFrame()#columns=['Frequency'])
    uns2uns_duration.index.name = 'Steps'
    
#    unseen2unseen = pd.DataFrame(columns = np.arange(1,frames_data.shape[1],1,dtype=np.int))
    for filename in glob.glob(resultsPath + '*/*/*-' + resultCategory + '.csv'):
#        if '608' in filename:
#            #skip ground truth data
#            continue
        videoName,networkName,_csvName = filename.replace(resultsPath,'').split('/')
        dataHeader = pd.MultiIndex.from_product([[networkName],cols],names=['Network','Data'])
        s2s_duration = update_duration_df_header(s2s_duration,dataHeader)
        uns2uns_duration = update_duration_df_header(uns2uns_duration,dataHeader)
        print(videoName,networkName)
        
        metrics_data = get_detection_metric_data(filename,visible_threshold=visible_threshold)
        frames_data = metrics_data.drop(metrics_data.columns[0:7],axis=1)
        
        for row_name,row_data in frames_data.iterrows():
            s2s_data = consecutive_occurrences(row_data,1) #seen values are 1
            s2s_duration = populate_duration_frequency(s2s_duration,s2s_data,networkName)
            
            uns2uns_data = consecutive_occurrences(row_data,0) #seen values are 1
            uns2uns_duration = populate_duration_frequency(uns2uns_duration,uns2uns_data,networkName)
#        print(row_name)#,s2s_data)
        
        
#        print(s2s_data)
#        s2s_data = s2s_data.value_counts() # count frequency of occurence of each detection
#        s2s_data = s2s_data.sort_index()
#        seen2seen.loc[row_name,s2s_data.index.values] = s2s_data
    return s2s_duration.sort_index(),uns2uns_duration.sort_index()

def update_duration_columns(df):
    networkNames = df.columns.get_level_values(0).unique()
    for network in networkNames:
        df[(network,'FreqxSteps')] = df[(network,'Frequency')].mul(df.index)
        df[(network,'PctFreq')] = df[(network,'Frequency')] / df[(network,'Frequency')].sum() * 100
        df[(network,'PctFreqxSteps')] = df[(network,'FreqxSteps')] / df[(network,'FreqxSteps')].sum() * 100
    return df

def bin_durations(df,binStart,binEnd,binWidth):
    binRanges = [(i,i+binWidth - 1) for i in np.arange(binStart,binEnd + 1,binWidth,dtype=np.int)] \
                    + [(binEnd + 1,df.index[-1])]
    binnedDF = pd.DataFrame(columns = df.columns)
    for binRange in binRanges:
        binMin,binMax = binRange
        indexMask = (df.index >= binMin) & (df.index <= binMax)
        rangeStr = ('-').join([str(i) for i in binRange])
        binnedDF.loc[rangeStr,:] = df.loc[indexMask,:].sum()
        
    return binnedDF
#def bar_plot(df,figName):
#    f = plt.figure(figsize=(6,4))
#    df.plot(kind='bar',colormap='inferno',stacked=True,ax=f.gca())
#    f.savefig(figName, bbox_inches='tight')
#def plot_binned_s2s_uns2uns_data(summary_analysis):
#    #probably unnecessary. Better to convert df to latex
#    for bincsv in ['*binned_s2s*.csv','*binned_uns2uns*.csv']:
#        for binName in glob.glob(summary_analysis + bincsv):
#            binnedDF = pd.read_csv(binName,header=[0,1],index_col=0,low_memory=False)
#            networkNames = binnedDF.columns.get_level_values(0).unique()
#            for ntwk in networkNames:
#                ntwkDF = binnedDF[ntwk]

def binned_s2s_uns2uns_data_to_latex(summary_analysis):
    subCols = ['f (%)','f x t (%)']
#    ntwks = ['yolo-128','yolo-224','mobileSSD-124','mobileSSD-220']
#    dataHeader = pd.MultiIndex.from_product([ntwks,subCols],names=['Network','t'])
    for bincsv in ['*binned_s2s*.csv','*binned_uns2uns*.csv']:
        binned_latexDF = None#pd.DataFrame()
        for binName in glob.glob(summary_analysis + bincsv):
            binnedDF = pd.read_csv(binName,header=[0,1],index_col=0,low_memory=False)
            binnedDFIndex = binnedDF.index.values
            #change the last index so it is uniform across networks
            binnedDFIndex[-1] = '> 50'
            binnedDF.index = binnedDFIndex
            networkNames = binnedDF.columns.get_level_values(0).unique()
            for ntwk in networkNames:
                print(ntwk)
                ntwkShort = shorten_network_name(ntwk)
                dataHeader = pd.MultiIndex.from_product([[ntwkShort],subCols])
                if binned_latexDF is None:
                    binned_latexDF = pd.DataFrame(columns = dataHeader,index = binnedDF.index)
#                    print(binned_latexDF)
                else:
                    binned_latexDF = binned_latexDF.reindex(columns = binned_latexDF.columns.union(dataHeader))
                binned_latexDF.loc[binnedDF.index,(ntwkShort,subCols[0])] = \
                    ['{:.0f} ({:.2f})'.format(d,pctd) for d,pctd in zip(binnedDF[ntwk]['Frequency'],binnedDF[ntwk]['PctFreq'])]
                binned_latexDF.loc[binnedDF.index,(ntwkShort,subCols[1])] = \
                    ['{:.0f} ({:.2f})'.format(d,pctd) for d,pctd in zip(binnedDF[ntwk]['FreqxSteps'],binnedDF[ntwk]['PctFreqxSteps'])]
                    
#                print(binnedDF[ntwk])
#                return
#                for tbin in binnedDF.index:
##                    print(binnedDF.loc[tbin,[(ntwk,'Frequency'),(ntwk,'PctFreq')]].values)
#                    binned_latexDF.loc[tbin,ntwk + '-f'] = \
#                        '{:.0f} ({:.2f})'.format(*binnedDF.loc[tbin,[(ntwk,'Frequency'),(ntwk,'PctFreq')]])
#                    binned_latexDF.loc[tbin,ntwk + '-ft'] = \
#                        '{:.0f} ({:.2f})'.format(*binnedDF.loc[tbin,[(ntwk,'FreqxSteps'),(ntwk,'PctFreqxSteps')]])
#            print(binName.replace('.csv','.tex'))
            binned_latexDF.to_latex(binName.replace('.csv','.tex'))
                        
def fine_grain_s2s_uns2uns_data(summary_analysis=None,computational_simulation=None,minT=1,maxT=20,plot=True):
    #fine grain analysis/histogram of s2s and uns2uns data to visualize their histograms
    dataDict = {'s2s':None,'uns2uns':None}
    for csvPattern in ['/*s2s_duration.csv', '/*uns2uns_duration.csv']:
        latexDF = None
        dataDF = None
        for csvData in glob.glob(summary_analysis + csvPattern):
            if 'binned' in csvData:#skip binned data
                continue
            csvDataDF = pd.read_csv(csvData,header=[0,1],index_col = 0, low_memory = False)
            csvDataDF = csvDataDF.loc[(minT <= csvDataDF.index) & (csvDataDF.index <= maxT),:]
            
            csvDataDF.drop(labels=['FreqxSteps','PctFreqxSteps'],axis=1,level=1,inplace=True)
#            print(csvDataDF)
            for ntwk in csvDataDF.columns.get_level_values(0).unique():
                print(ntwk)
                ntwkShort = shorten_network_name(ntwk)
                if latexDF is None:
                    latexDF = pd.DataFrame(columns = [ntwkShort], index = csvDataDF.index)
                    dataDF = pd.DataFrame(columns = [ntwkShort], index = csvDataDF.index)
                else:
                    latexDF = latexDF.reindex(columns = latexDF.columns.union([ntwkShort]))
                    latexDF = latexDF.reindex(index = latexDF.index.union(csvDataDF.index))
                    
                    dataDF = dataDF.reindex(columns = dataDF.columns.union([ntwkShort]))
                    dataDF = dataDF.reindex(index = dataDF.index.union(csvDataDF.index))
#                    latexDF = latexDF.join(pd.DataFrame(columns=[ntwkShort],index = csvDataDF.index))
#                print(latexDF.index,csvDataDF.index)
                dataDF.loc[csvDataDF.index,ntwkShort] = csvDataDF[ntwk]['PctFreq']
                latexDF.loc[csvDataDF.index,ntwkShort] = \
                    ['{:.0f} ({:.2f})'.format(d,pctd) for d,pctd in zip(csvDataDF[ntwk]['Frequency'],csvDataDF[ntwk]['PctFreq'])]
    
            
            #output filename
            fname = csvData.replace('.csv',f'_range{minT}-{maxT}')
            #save as latex table
            latexDF.to_latex(fname + '.tex',escape=False)
            
#            print(csvData)
            if 'TPandFN' in csvData:
#                print(dataDF)
                if 's2s' in csvPattern:
                    dataDict['s2s'] = dataDF
                else:
                    dataDict['uns2uns'] = dataDF
            if plot:
                #plot data
                if 's2s' in csvPattern:
                    dataDict['s2s'] = dataDF
                    xlabel = 'No. of Consecutive seen2seen'
                else:
                    dataDict['uns2uns'] = dataDF
                    xlabel = 'No. of Consecutive unseen2unseen'
                #rename column names
                pctFreqDF = csvDataDF.xs('PctFreq',level=1,drop_level=False,axis=1)
                pctFreqDF.columns = [shorten_network_name(i) for i in pctFreqDF.columns.droplevel(1)]
                plot_metric_data(pctFreqDF,fname + '.png',ylim=[0,100],
                                 ylabel = 'Percentage from Total Data',xlabel = xlabel)
            
    return dataDict
    #            csvDataDF  = csvDataDF.xs(['Frequency','PctFreq'],axis=1,drop_level=False,level=1)
#            print(csvDataDF.index)
#            return latexDF
        
def compare_fine_grain(experiments,comp_model,outputFolder,compCSV=None):
    experimentDict = fine_grain_s2s_uns2uns_data(experiments,plot=False)
    comp_modelDict = fine_grain_s2s_uns2uns_data(comp_model,plot=False)
#    print(experimentDict.shape,comp_modelDict.shape)
    dataHeader = pd.MultiIndex.from_product([experimentDict['s2s'].columns,['Experiment','Comp_Model']])
    
#    return experimentDict,comp_modelDict
    for category in experimentDict.keys():
        print(category)
        if 's2s' in category:
            xlabel = 'No. of Consecutive seen2seen'
        else:
            xlabel = 'No. of Consecutive unseen2unseen'
        
        dfData = pd.DataFrame(columns = dataHeader)
        fname = outputFolder + '/compare-' + ntpath.basename(experiments) + '-and-' + \
                    ntpath.basename(comp_model) + '-' + category
        for col in experimentDict['s2s'].columns:
            print('\t',col)
             
                    
            dfData[(col,'Experiment')] = experimentDict[category][col]
            dfData[(col,'Comp_Model')] = comp_modelDict[category][col]
            
            plot_metric_data(dfData[col],fname + '-' + col + '.png',ylim=[0,50],xlabel=xlabel
                             ,ylabel='Percentage from Total Data')
        dfData.to_csv(fname + '.csv')
        dfData.to_latex(fname + '.tex',escape=False)
    
            
           
            
def process_s2s_uns2uns_data(resultsPath,summary_analysis,categories=['TPandFN','FPdata'],visible_threshold=1):
    for resultCategory in categories:
        print(resultCategory)
        s2s_duration,uns2uns_duration = process_consecutive_occurrences(resultsPath,resultCategory,visible_threshold=visible_threshold)
        s2s_duration = update_duration_columns(s2s_duration)
        s2s_duration.to_csv(summary_analysis + resultCategory + '_s2s_duration.csv')
        binned_s2s_duration = bin_durations(s2s_duration,1,50,10)
        binned_s2s_duration.to_csv(summary_analysis + resultCategory + '_binned_s2s_duration.csv')
        
        uns2uns_duration = update_duration_columns(uns2uns_duration)
        uns2uns_duration.to_csv(summary_analysis + resultCategory + '_uns2uns_duration.csv')
        binned_uns2uns_duration = bin_durations(uns2uns_duration,1,50,10)
        binned_uns2uns_duration.to_csv(summary_analysis + resultCategory + '_binned_uns2uns_duration.csv')
        
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
def shorten_network_name(networkName):
    if 'mobile' in networkName:
        ntwk = 'mSSD-'
    elif 'yolov3-tiny' in networkName:
        ntwk = 'yolov3-tiny-'
    elif 'yolov3-litter' in networkName:
        ntwk = 'yolov3-'
    else:
        return networkName
    
    ntwk += networkName.replace('iSz','').split('-')[-1]
    return ntwk

def plot_metric_data(df,filename,legend=True,ylim=[0,1],ylabel=None,xlabel=None):
    f = plt.figure()
    ax = f.gca()
    df.index = np.arange(1,df.shape[0] + 1,1,dtype=np.int)
    df.plot(kind='line',style='-o',legend=False,ax = ax,ylim=ylim)#,xticks=np.linspace(1,df.shape[0],1,dtype=np.int))
#    print(np.arange(0,df.shape[0],1,dtype=np.int))
#    ax.set_xticks(np.arange(1,df.shape[0] + 1,1,dtype=np.int))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='best',ncol=2)
    if ylabel is not None:
        plt.ylabel(ylabel,fontweight='bold')
    if xlabel is not None:
        plt.xlabel(xlabel,fontweight='bold')
        
    f.savefig(filename,bbox_inches='tight')
    
def summary_csv_data_to_latex(filename):
    df = pd.read_csv(filename + '.csv',header=[0,1],index_col=0)
    metricSUM = []
    metricMean = []
    for metric in df.columns.get_level_values(1).unique():
        if 'P_' in metric:
            #average probabilities
            metricMean.append(metric)
        elif 'pruned' not in metric:
            #other metric data should be summed up
            metricSUM.append(metric)
    latexDF = pd.DataFrame(index= metricSUM + metricMean)
    for network in df.columns.get_level_values(0).unique():
        if '608' in network:
            continue
        print(network)
        
        ntwk = shorten_network_name(network)
        networkDF = df[network]
        
        latexDF.loc[metricSUM,ntwk] = networkDF[metricSUM].sum()
        latexDF.loc[metricMean,ntwk] = ['${:.4f} \pm {:.4f}$'.format(m,s) \
                 for m,s in zip(networkDF[metricMean].mean(),networkDF[metricMean].std())]
    latexDF.to_latex(filename + '.tex',escape=False)
        
def visualize_summary_csv_data(filename):
    df = pd.read_csv(filename + '.csv',header=[0,1],index_col=0)
    gt_yolo608 = 'yolov3-litter_10000-th0p0-nms0p0-iSz608'
    if gt_yolo608 in df.columns:
        unique_litterDF = df[gt_yolo608]
    #    print(unique_litterDF.columns)
        unique_litterDF.columns = [i.replace('nlitter_','') for i in unique_litterDF.columns]
        if 'TPandFN' in filename:
            plot_metric_data(unique_litterDF,filename + 'nlitter.png',ylim=[0,unique_litterDF.max().max() * 1.1],
                                        ylabel = 'No. of Unique Litters',xlabel='Video Index')
            print(unique_litterDF.sum())
        
    for metric in df.columns.get_level_values(1).unique():
        if 'P_' in metric:#only instrested in plotting probabilities
            metricDF = df.xs(metric,level=1,drop_level=False,axis=1)
            metricDF.columns = [shorten_network_name(i) for i in metricDF.columns.droplevel(1)]
            plot_metric_data(metricDF,filename + '-' + metric + '.png',\
                             ylabel = metric, xlabel='Video Index')
    
#    return df
def generate_extra_summary_columns(df):
    df2 = pd.DataFrame(columns=['firstState','unseen_uns2s','unseen_uns2uns',\
                                'seen_uns2uns','seen_uns2s','unseen_s2uns',\
                                'unseen_s2s','seen_s2uns','seen_s2s'])
    for row in df.index:
        rowData = df.loc[row,:].dropna()
        df2.loc[row,'firstState'] = rowData[0]
        
        seenData = rowData[rowData == 1]#information about indexes where litter is seen
        if len(seenData) > 0:
            firstSeenIdx = seenData.index[0]
            
            df2.loc[row,'unseen_uns2s'] = 1 if df2.loc[row,'firstState'] == 0 else 0
            totalUnseen_uns = len(rowData[:firstSeenIdx]) - 1
            df2.loc[row,'unseen_uns2uns'] = totalUnseen_uns - 1 if totalUnseen_uns > 0 else 0 #minus the time-step it was seen
            
                
            #process uns2uns and uns2s for duration that litter has been seen
            seenData = rowData[firstSeenIdx:]
#            print(seenData[0])
#            print(firstSeenIdx,consecutive_occurrences(seenData,0).sum(),rowData)
            totalUnseen = consecutive_occurrences(seenData,0)
            
            df2.loc[row,'seen_uns2uns'] = totalUnseen.sum() - len(totalUnseen) if len(totalUnseen) > 0 else 0
            df2.loc[row,'seen_uns2s'] = sum(seenData.diff().dropna().astype(np.int) == 1)
            
            if len(seenData.index) > 1:#if there is at least one detection data after first detection of the specific litter
                if seenData.iloc[1] == 1:
                    df2.loc[row,'unseen_s2uns'] = 0
                    df2.loc[row,'unseen_s2s'] = 1
                elif seenData.iloc[1] == 0:
                    df2.loc[row,'unseen_s2uns'] = 1
                    df2.loc[row,'unseen_s2s'] = 0
                else:
                    print('ERROR: expected 0 or 1 but got: ',seenData.iloc[1])
                
                totalSeen = consecutive_occurrences(seenData.iloc[1:],1)
                df2.loc[row,'seen_s2s'] = totalSeen.sum() - len(totalSeen) if len(totalSeen) > 0 else 0
                df2.loc[row,'seen_s2uns'] = sum(seenData.iloc[1:].diff().dropna().astype(np.int) == -1)
            else:
                df2.loc[row,'unseen_s2uns'] = 0
                df2.loc[row,'unseen_s2s'] = 0
                df2.loc[row,'seen_s2uns'] = 0
                df2.loc[row, 'seen_s2s'] = 0
        else:
            df2.loc[row,'seen_uns2uns'] = 0
            df2.loc[row,'seen_uns2s'] = 0
            df2.loc[row,'unseen_uns2s'] = 0
            df2.loc[row,'unseen_uns2uns'] = len(rowData) - 1 if len(rowData) > 0 else 0#was seen the whole time
            
            df2.loc[row,'unseen_s2uns'] = 0
            df2.loc[row,'unseen_s2s'] = 0
            df2.loc[row,'seen_s2uns'] = 0
            df2.loc[row, 'seen_s2s'] = 0
     
    return df2

def summarise_csv_data(resultsPath,resultCategory,visible_threshold=1):
    '''
    Analyse litters data based on video names
    '''
    summaryDF = pd.DataFrame()
    summaryCols = ['seen2seen','seen2unseen','P_s2s',\
                'unseen2seen','unseen2unseen','P_u2s',\
                'seen','unseen','visible','P_visible']
    visibleData = None
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
                dataHeader = pd.MultiIndex.from_product([[networkName],\
                            ['nlitter_pruned','nlitter_unpruned']],names=['Network','Data'])
                if len(summaryDF.columns) == 0:
                    summaryDF = pd.DataFrame(columns=dataHeader)
                    # self.trackerDF.columns.names
                else:
                    # print(self.trackerDF.columns.union(dataHeader))
                    summaryDF = summaryDF.reindex(columns=summaryDF.columns.union(dataHeader))
                
                csvDF = pd.read_csv(csvFile + '-GT-unpruned.csv',header=[0,1],index_col=0,
                                    low_memory=False,nrows=10)
                litterIDs = csvDF.columns.get_level_values(0).unique()
                summaryDF.loc[videoName,(networkName,'nlitter_unpruned')] = len(litterIDs)
                
                csvDF = pd.read_csv(csvFile + '-GT-pruned.csv',header=[0,1],index_col=0,
                                    low_memory=False,nrows=10)
                litterIDs = csvDF.columns.get_level_values(0).unique()
                summaryDF.loc[videoName,(networkName,'nlitter_pruned')] = len(litterIDs)
                continue
            csvFile = csvFile + '-detection-' + resultCategory + '.csv'
            rawDataDF = get_detection_metric_data(csvFile,visible_threshold=visible_threshold)
            
            
            
            #logged information from raw data
            csvDF = rawDataDF.loc[:,'seen2seen':'visible']
#            return csvDF,rawDataDF
            #from raw data extract state (i.e. seen/unseen) for first occurrence in GT, 
            #seen_uns2s, seen_uns2uns, unseen_uns2s, unseen_uns2uns
            #DATA
            #return csvDF,rawDataDF
            extraDataDF = generate_extra_summary_columns(rawDataDF.drop(labels=csvDF.columns,axis=1))
            csvDF = extraDataDF.join(csvDF)
            
            summaryCols = ['nlitter'] + list(extraDataDF.columns.values) + \
                                ['P_s_u2s','P_uns_u2s','P_s_s2s','P_uns_s2s'] + summaryCols
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
            csvDFsummary['nlitter'] = rawDataDF.shape[0]#number of rows is number of unique litter
            
            #old metrics data
            csvDFsummary['P_s2s'] = \
                csvDFsummary['seen2seen'] / (csvDFsummary['seen2seen'] + csvDFsummary['seen2unseen'])
            csvDFsummary['P_u2s'] = \
                csvDFsummary['unseen2seen'] / (csvDFsummary['unseen2seen'] + csvDFsummary['unseen2unseen'])
            csvDFsummary['P_visible'] = \
                csvDFsummary['seen'] / (csvDFsummary['visible'])
            
            #extra_data metrics data
            if 'seen_uns2s' not in csvDFsummary.index:
                
                return csvDFsummary
            csvDFsummary['P_s_u2s'] = \
                csvDFsummary['seen_uns2s'] / (csvDFsummary['seen_uns2s'] + csvDFsummary['seen_uns2uns'])
            csvDFsummary['P_uns_u2s'] = \
                csvDFsummary['unseen_uns2s'] / (csvDFsummary['unseen_uns2s'] + csvDFsummary['unseen_uns2uns'])
            
            csvDFsummary['P_s_s2s'] = \
                csvDFsummary['seen_s2s'] / (csvDFsummary['seen_s2s'] + csvDFsummary['seen_s2uns'])
            csvDFsummary['P_uns_s2s'] = \
                csvDFsummary['unseen_s2s'] / (csvDFsummary['unseen_s2s'] + csvDFsummary['unseen_s2uns'])
            summaryDF.loc[videoName,dataHeader] = csvDFsummary.loc[summaryCols].values
#            summaryDF.merge(csvDF.sum(),how='left',on=videoName,left_on=networkName)
#            csvDF = csvDF
#            return csvDF
#        print(f"nlitter {len(rawDataDF['visible'].values)}")
        if visibleData is None:
            visibleData = list(rawDataDF['visible'].values)
        else:
            visibleData = visibleData + list(rawDataDF['visible'].values)
    return summaryDF,visibleData

def get_network_data(resultsPath,networks):
    first_last_columns = ['x1','x2','y1','y2']
    metrics_columns = ['seen2seen','seen2unseen','unseen2seen','unseen2unseen','seen','unseen','visible']
    for networkName in networks:
        print(networkName)
        print()
        print()
        df = None
        for videoFolder in glob.glob(resultsPath + f'*/*{networkName}'):
            videoName,network = videoFolder.replace(resultsPath,'').split('/')
            print(videoName)
            detectionDF = pd.read_csv(videoFolder + '/' + videoName + '-' + networkName + '-detection.csv',
                                      header = [0,1], index_col=0,low_memory=False)
            tp_fn_df = pd.read_csv(videoFolder + '/' + videoName + '-' + networkName + '-detection-TPandFN.csv',
                                   index_col=0,header=0)
            index = pd.MultiIndex.from_product([[videoName],tp_fn_df.index.values])
                
            if df is None:
    #            print (tp_fn_df.loc[:,metrics_columns].values)
                cols = pd.MultiIndex.from_product([['metrics_columns'],metrics_columns])
                df = pd.DataFrame(index=index,columns=cols)
                
                
                newcols = pd.MultiIndex.from_product([['firstSeen','lastSeen'],first_last_columns])
                df = df.reindex(columns = df.columns.union(newcols,sort=False),    )
                
                
            else:
                df = df.reindex(index = df.index.union(index,sort=False))
            df.loc[videoName,'metrics_columns'] = tp_fn_df.loc[:,metrics_columns].values
            for lit in tp_fn_df.index:
                
                litDF = detectionDF[lit].dropna()
                litDF = litDF.loc[litDF['info'] == 'TP',first_last_columns]
                if litDF.shape[0] > 0:
                    first = litDF.index[0]
                    last = litDF.index[-1]
                    df.loc[(videoName,lit),'firstSeen'] = litDF.loc[first,first_last_columns].values
                    df.loc[(videoName,lit),'lastSeen'] = litDF.loc[last,first_last_columns].values
    #            df.loc[(videoName,lit),'metrics_columns'] = tp_fn_df.loc[lit,metrics_columns].values
        ntwk = shorten_network_name(networkName)
        os.makedirs('../data/simplified_data', exist_ok=True)
        df.to_csv('../data/simplified_data/' + ntwk + '.csv')
        print()
        print()
        print()
            
            
            
def get_groundtruth_data(resultsPath):
    df = None
    columns = ['x1','x2','y1','y2']
    
    for videoFolder in glob.glob(resultsPath + '/*/*608*'):
        videoName,networkName = videoFolder.replace(resultsPath,'').split('/')
        print(videoName)
        csvDF = pd.read_csv(videoFolder + '/' + videoName + '-' + networkName + '-GT-pruned.csv',
                            header = [0,1], index_col = 0, low_memory = False)
        litterIDs = csvDF.columns.get_level_values(0).unique()
        for lit in litterIDs:
            litDF = csvDF[lit].dropna()
            first = litDF.index[0]
            last = litDF.index[-1]
            visible = litDF.shape[0]
            newIndex = pd.MultiIndex.from_product([[videoName],[lit]])
            newColumns = pd.MultiIndex.from_product([['firstAppearance','lastAppearance'],columns])
            if df is None:
                df = pd.DataFrame(index = newIndex, columns = newColumns)
                
            else:
                
                df = df.reindex(index = df.index.union(newIndex,sort=False))
            df.loc[(videoName,lit),'firstAppearance'] = litDF.loc[first,columns].values
            df.loc[(videoName,lit),'lastAppearance'] = litDF.loc[last,columns].values
            df.loc[(videoName,lit),'visible'] = visible
#            return df,litDF
#        if 'GOPR9069' in videoName:
    df.to_csv(f'../data/simplified_data/yolo_608.csv')
    return df

def summarise_csv_data_litters(resultsPath,resultCategory,visible_threshold=1):
    '''
    Analyse litters data based on each litter
    '''
    summaryDF = None#pd.DataFrame()
#    summaryCols = ['seen2seen','seen2unseen','P_s2s',\
#                'unseen2seen','unseen2unseen','P_u2s',\
#                'seen','unseen','visible','P_visible']
#    visibleData = None
    for videoFolder in glob.glob(resultsPath + '*'):
        videoName = ntpath.basename(videoFolder)
        print(videoName,videoFolder)
        for networkFolder in glob.glob(videoFolder + '/*'):
            networkName = ntpath.basename(networkFolder)
            print(networkName)
            csvFile = networkFolder + '/' + videoName + '-' + networkName
            if '608' in networkName:
                continue
            csvFile = csvFile + '-detection-' + resultCategory + '.csv'
            rawDataDF = get_detection_metric_data(csvFile,visible_threshold=visible_threshold)
            csvDF = rawDataDF.loc[:,'seen2seen':'visible']
            if csvDF.shape[0] == 0:
                continue
#            extraDataDF = generate_extra_summary_columns(rawDataDF.drop(labels=csvDF.columns,axis=1))
#            csvDF = extraDataDF.join(csvDF)
            csvDF.loc[:,'P_s2s'] = csvDF['seen2seen'] / (csvDF['seen2seen'] + csvDF['seen2unseen'])
            csvDF.loc[:,'P_u2s'] = csvDF['unseen2seen'] / (csvDF['unseen2seen'] + csvDF['unseen2unseen'])
            csvDF.loc[:,'P_visible'] = csvDF['seen'] / csvDF['visible']
            if summaryDF is None:
                summaryDF = pd.DataFrame(index = pd.MultiIndex.from_product([[videoName],csvDF.index.values]),
                                       columns = pd.MultiIndex.from_product([[networkName],csvDF.columns.values]))
                summaryDF.loc[videoName,networkName].update(csvDF)
            else:
                newindex = pd.MultiIndex.from_product([[videoName],csvDF.index.values])
                newcolumns = pd.MultiIndex.from_product([[networkName],csvDF.columns.values])
                newDF = pd.DataFrame(index=newindex,columns=newcolumns)
                newDF.loc[videoName,networkName].update(csvDF)
                if videoName in summaryDF.index.get_level_values(0):
                    summaryDF = summaryDF.reindex(columns = summaryDF.columns.union(newcolumns))
#                    summaryDF = pd.concat([summaryDF,newDF],axis=1,join='inner')
                    summaryDF = summaryDF.combine_first(newDF)
                else:
                    summaryDF = summaryDF.reindex(index = summaryDF.index.union(newindex))
                    summaryDF = summaryDF.combine_first(newDF)
#                    summaryDF = summaryDF.append(newDF)
                    
#                summaryDF

    return summaryDF
            
def close2frame_border_mask(fwidth,fheight,edgewidth,horizon):
    center,radius,start_angle,end_angle = utils.findCircle(*horizon)
    maskDF = pd.DataFrame(index=np.arange(1,fheight+1,1,dtype=np.int),
                          columns=np.arange(1,fwidth+1,1,dtype=np.int),dtype=np.bool)
    maskDF.loc[:,:] = False
    
    for x in maskDF.columns:
#        theta = np.arccos((x - center[0])/float(radius))
        radiusY = center[1] -  np.sqrt(np.square(radius) - np.square(x - center[0]))  #radius * np.sin(theta) - fheight#
#        print(x)#,radiusY,theta)
        if True and (x <= edgewidth or x >= maskDF.columns[-1] - edgewidth):
            ymax = maskDF.index[-1]
        else:
            ymax = np.rint(radiusY + edgewidth)#/2.0)
        ymin = maskDF.index[0]#np.rint(radiusY)# - edgewidth/2.0)
        
        maskDF.loc[ymin:ymax,x] = True
    
    sns.heatmap(maskDF.astype(np.int),vmax=1)
    return maskDF
def pct_close_to_frame_border(filename,maskDF=None):
    if maskDF is None:
        maskDF = close2frame_border_mask(960,540,10,[18,162,494,59,937,143])
    first_appearanceDF = pd.read_csv(filename,index_col=0,header=0)
#    heatmap(first_appearanceDF,filename.replace('.csv','.png'),vmax=1,colorbar=False)
    othersDF = pd.DataFrame(index=maskDF.index,columns=maskDF.columns)
    othersDF.loc[:,:] = 0
    
    total_appearances = first_appearanceDF.sum().sum()
    total_close_to_frame_border = 0
    for col in first_appearanceDF.columns:
        print(col)
        total_close_to_frame_border += first_appearanceDF.loc[maskDF[int(col)],col].sum()
        othersDF.loc[~maskDF[int(col)],int(col)] = first_appearanceDF.loc[~maskDF[int(col)],col]
    pct = float(total_close_to_frame_border)/total_appearances * 100
    
    sns.heatmap(othersDF,vmax=1,cbar=False,yticklabels=[],xticklabels=[])
    print(total_appearances,total_close_to_frame_border,pct)
    return othersDF,first_appearanceDF#,total_appearances,total_close_to_frame_border,pct

def apply_left_right_bottom_threshold(filename,last_seen_threshold=30,last_seen_min_y=300,
                                      threshold=1,horizon=[18,162,494,59,937,143]):
    df = pd.read_csv(filename,index_col=[0,1],header=[0,1])
    df = df.loc[df['visible'].iloc[:,0] >= threshold,:]
    
    df.loc[:,('firstAppearance','cx')] = df.loc[:,[('firstAppearance','x1'),('firstAppearance','x2')]].mean(axis=1).astype(np.int)
    df.loc[:,('firstAppearance','cy')] = df.loc[:,[('firstAppearance','y1'),('firstAppearance','y2')]].mean(axis=1).astype(np.int)
    
    df.loc[:,('lastAppearance','cx')] = df.loc[:,[('lastAppearance','x1'),('lastAppearance','x2')]].mean(axis=1).astype(np.int)
    df.loc[:,('lastAppearance','cy')] = df.loc[:,[('lastAppearance','y1'),('lastAppearance','y2')]].mean(axis=1).astype(np.int)
    left = (df.loc[:,('lastAppearance','cx')] <= last_seen_threshold) & (df.loc[:,('lastAppearance','cy')] >= last_seen_min_y)
    right = (df.loc[:,('lastAppearance','cx')] >= (960 - last_seen_threshold)) & (df.loc[:,('lastAppearance','cy')] >= last_seen_min_y)
    bottom = df.loc[:,('lastAppearance','cy')] >= (540 - last_seen_threshold)
    
    center,radius,start_angle,end_angle = utils.findCircle(*horizon)
    horizon_start = df['firstAppearance'].apply(lambda  x: \
            abs(np.linalg.norm(x[['cx','cy']]-center) - radius) <= 30,axis=1)
    df = df.loc[(left | right | bottom) & horizon_start ,:]
    return df

def analyse_first_last_visible(filename,output_folder,threshold=1,last_seen_threshold=30,last_seen_min_y = 300):
    figname = output_folder + '/{}-threshold_{}-nlitter_{}-last_seen_threshold_{}-last_seen_min_y_{}'
    appearanceDF = pd.DataFrame(index=np.arange(1,541,1,dtype=np.int),
                            columns=np.arange(1,961,1,dtype=np.int))
    
    

    df = apply_left_right_bottom_threshold(filename)    
    heatmapData = {}
    for appearance in ['firstAppearance','lastAppearance']:
        print(appearance)
        appearanceDF.loc[:,:] = 0
        for i in df.index:
#            print(i)
            cx = df.loc[i,(appearance,'cx')]
            cy = df.loc[i,(appearance,'cy')]
            if cx in appearanceDF.columns and cy in appearanceDF.index:
                appearanceDF.loc[cy,cx] += 1
        appearanceDF = appearanceDF.astype(np.int)
        heatmapData[appearance] = appearanceDF.copy()
#        return appearanceDF
        heatmap(appearanceDF,figname.format(appearance,threshold,df.shape[0],last_seen_threshold,last_seen_min_y),
                    vmax=1,colorbar=False)
    f2 = plt.figure(figsize=(16,9))
    ax = f2.gca()
    for i in df.index:
        x = df.loc[i,[('firstAppearance','cx'),('lastAppearance','cx')]]
        y = df.loc[i,[('firstAppearance','cy'),('lastAppearance','cy')]]
        plt.plot(x,y,color=(0,0,0,0.2))
    ax.set_ylim(bottom=1,top=540)
    ax.set_xlim(left=1,right=960)
    ax.invert_yaxis()
    plt.savefig(figname.format('first_to_last',threshold,df.shape[0],last_seen_threshold,last_seen_min_y),bbox_inches='tight')
    return df,heatmapData

def create_frameDim_DF():
    frame = pd.DataFrame(index=np.arange(1,541,1,dtype=np.int),
                                     columns=np.arange(1,961,1,dtype=np.int))
    frame.loc[:,:] = 0
    return frame
def heatmap(df,figName,vmax,colorbar=True):
    df.to_csv(figName + '.csv')
    f = plt.figure(figsize=(10,6))
    sns.heatmap(df,annot=False,xticklabels=[],yticklabels=[],ax=f.gca(),vmax=vmax,cbar=colorbar)
    f.savefig(figName + '.png', bbox_inches='tight')
    
def bbox_heatdata(resultsPath,summary_analysis,
                  networkPatterns=['*mobile*220*','*yolo*128*','*yolo*224*','*mobile*124*',],
                  centre = True):
    heatMax = None
    dataDict = {}
    for ntwk in networkPatterns:
        if '608' not in ntwk:
            dataDict['TP'] = create_frameDim_DF()
            dataDict['FN'] = create_frameDim_DF()
            dataDict['FP'] = create_frameDim_DF()
        else:#ignore groundtruth network for now
            continue
        box_centresDF_csv = None
        
        for videoFolder in glob.glob(resultsPath + '/*/' + ntwk):
            videoName,networkName = videoFolder.replace(resultsPath,'').split('/')
            if box_centresDF_csv is None:
                box_centresDF_csv = summary_analysis + networkName + '-bbox_heatdata'
            
            csvDF = pd.read_csv(videoFolder + '/' + videoName + '-' + networkName +'-detection.csv', # '-GT-pruned.csv',
                                header = [0,1], index_col = 0, low_memory = False)
            litterIDs = csvDF.columns.get_level_values(0).unique()
            
            for lit in litterIDs:
#                if 'F' in lit:
#                    continue#ignore False positives for now
#                csvDFMask = csvDF[(lit,'info')] == 'TP'
#                boxCentre = csvDF.loc[csvDFMask,lit]
    #            boxCentre = csvDF[lit].dropna(axis=0,how='all')
                
    #            cx = boxCentre[['x1','x2']].mean(axis=1).astype(int)#.values[0]
    #            cy = boxCentre[['y1','y2']].mean(axis=1).astype(int)#.values[0]
    #            print(cx,cy)
                bbox = csvDF[lit].dropna(axis=0,how='all')#drop nan rows
                
                for i,b in bbox.iterrows():
                    x1,x2,y1,y2 = b[['x1','x2','y1','y2']].astype(int)
                    dictKey = b['info']
                    if 'inter,FP' in dictKey:
                        #do not do anything when it is interpolated FP
                        continue
                    if 'FP' in dictKey:
                        dictKey = 'FP'
                    
                        
    #                print(b)
                    if centre:
                        cy,cx = np.rint(float(y1+y2)/2.0),np.rint(float(x1+x2)/2.0)
                        if 0 < cy <= 540 and 0 < cx <= 960:
                            dataDict[dictKey].loc[cy,cx] += 1
                    else:
                        if 0<x1<=960 and 0<x2<=960 and 0<y1<=960 and 0<y2<=960:
                            dataDict[dictKey].loc[y1:y2,x1:x2] += 1
    #            for y,x in zip(cy,cx):
    #                if y not in box_centresDF.index or x not in box_centresDF.columns:
    #                    continue
    #                box_centresDF.loc[y,x] += 1
    #            print(firstOccurrence)
                
            print(videoFolder)
        #sns.heatmap(box_centresDF,annot=False,xticklabels=[],yticklabels=[])
    #    sns.heatmap(box_centresDF,annot=False,xticklabels=[],yticklabels=[],vmax=1)
        if heatMax is None:
            heatMax = dataDict['TP'].max().max()
            if heatMax == 0:
                heatMax = None
            
        dataDict['FP'].to_csv(box_centresDF_csv + '-FP.csv')
        heatmap(dataDict['FP'],box_centresDF_csv + '-FP.png',heatMax)
        
        dataDict['TP'].to_csv(box_centresDF_csv + '-TP.csv')
        heatmap(dataDict['TP'],box_centresDF_csv + '-TP.png',heatMax)
        
        dataDict['FN'].to_csv(box_centresDF_csv + '-FN.csv')
        heatmap(dataDict['FN'],box_centresDF_csv + '-FN.png',heatMax)
#    return dataDict
def add_metrics_data(columns,df):
    metricsData = pd.DataFrame(columns = columns)
    for litter in df.index:
        rowData = df.loc[litter,:].dropna()
        metricsData.loc[litter,'visible'] = len(rowData)
        metricsData.loc[litter,'seen'] = sum(rowData == 1)
        metricsData.loc[litter,'unseen'] = sum(rowData == 0)
        
        seenFreqs = consecutive_occurrences(rowData,1)
        metricsData.loc[litter,'seen2seen'] = seenFreqs.sum() - len(seenFreqs) if len(seenFreqs) > 0 else 0
        
        unseenFreqs = consecutive_occurrences(rowData,0)
        metricsData.loc[litter,'unseen2unseen'] = unseenFreqs.sum() - len(unseenFreqs) if len(unseenFreqs) > 0 else 0
        
        stateChange = rowData.diff().dropna().astype(np.int)
        metricsData.loc[litter,'seen2unseen'] = sum(stateChange == -1) #change from seen (1) to unseen (0) i.e. 0 - 1 = -1
        metricsData.loc[litter,'unseen2seen'] = sum(stateChange == 1) #change from unseen (0) to seen (1) i.e. 1 - 0 = 1
    
    return metricsData.join(df)
        
        
def resample_experiment_data(resultsPath,rate,framerate=50,visible_threshold=1):
    processingTime = framerate / float(rate)
    
    if resultsPath[-1] == '/':
        resultsPath = resultsPath[:-1]
    outputPath = f'{resultsPath}-{rate}Hz-ge{visible_threshold}/{ntpath.basename(resultsPath)}'
    for videoFolder in glob.glob(resultsPath + '/*'):
        videoName = ntpath.basename(videoFolder)
        print(videoName)
#        if 'GOPR9061' != videoName:
#            continue
        for networkFolder in glob.glob(videoFolder + '/*'):
            networkName = ntpath.basename(networkFolder)
            
            if '608' in networkName:#ignore ground truth data/folder
                continue
            for resultCategory in ['TPandFN']:#,'FPdata':
                print(f'\t{networkName}')
                
                csvFile = videoName + '-' + networkName + '-detection-' + resultCategory + '.csv'
                rawDataDF = get_detection_metric_data(f'{networkFolder}/{csvFile}',visible_threshold=visible_threshold)
#                print(rawDataDF.columns)
                cols2drop = rawDataDF.loc[:,'seen2seen':'visible'].columns
                rawDataDF.drop(labels=cols2drop, axis=1,inplace=True)
                frames = np.ceil(np.arange(0,rawDataDF.columns.shape[0], processingTime)).astype(np.int)
                frames = frames[frames < rawDataDF.columns.shape[0]]#remove all index that will make it larger than number of frames index
#                if 'iSz128' in networkName and resultCategory is 'FPdata':
#                    return frames,rawDataDF.columns
#                return frames
                os.makedirs(f'{outputPath}/{videoName}/{networkName}/' ,exist_ok=True)
                filteredDF = add_metrics_data(cols2drop,rawDataDF.iloc[:,frames])
                filteredDF.to_csv(f'{outputPath}/{videoName}/{networkName}/{csvFile}' )
#                return filteredDF
def analyse_resultPath_and_summarize(resultPath,summaryPath,visible_threshold=1):
    summaryPath = f'{summaryPath}-ge_threshold-{visible_threshold}/'
    print(summaryPath)
#    input('>')
    os.makedirs( summaryPath,exist_ok=True)
    for resultCategory in ['TPandFN']:#,'FPdata'
        summaryDF,visibleData = summarise_csv_data(resultPath,resultCategory=resultCategory,visible_threshold=visible_threshold)
        summaryDF.to_csv(f'{summaryPath}/{resultCategory}_metrics_data.csv')
        visibleData.sort()
        if resultCategory == 'TPandFN':
            visibleDataDF = pd.DataFrame(Counter(visibleData),index=['Frequency']).T
        
            visibleDataDF.to_csv(f'{summaryPath}/{resultCategory}_visibleData.csv')
        summary_csv_data_to_latex(f'{summaryPath}/{resultCategory}_metrics_data')
        visualize_summary_csv_data(f'{summaryPath}/{resultCategory}_metrics_data')
        process_s2s_uns2uns_data(resultPath,f'{summaryPath}/',categories=[resultCategory],visible_threshold=visible_threshold)
    
    fine_grain_s2s_uns2uns_data(summary_analysis=f'{summaryPath}/')
    generate_plot_for_visibleData(f'{summaryPath}/')
def metrics_data_breakdown(break_down,col,metric_name,df,step=0.1):
    start = 0
    count = df.loc[(df[metric_name] == 0) | df[metric_name].isna(),metric_name].shape[0]
    pct = float(count) / df.shape[0] * 100
    break_down.loc[str(0),col] = f'{count} ({pct:02.2f}\%)'
    while start < 1:
        stop = round(start + step,1)
        if stop < 1:
            count = df.loc[(df[metric_name] > start) & (df[metric_name] <= (stop)),metric_name].shape[0]
            pct = float(count) / df.shape[0] * 100
            break_down.loc[f'{start}<i<={stop}',col] = f'{count} ({pct:02.2f}\%)'
        else:
            count = df.loc[(df[metric_name] > start) & (df[metric_name] < (stop)),metric_name].shape[0]
            pct = float(count) / df.shape[0] * 100
            break_down.loc[f'{start}<i<{stop}',col] = f'{count} ({pct:02.2f}\%)'
        start = stop
    count = df.loc[df[metric_name] == 1,metric_name].shape[0]
    pct = float(count) / df.shape[0] * 100
    break_down.loc[str(1),col] = f'{count} ({pct:02.2f}\%)'
    
    return break_down


def resample_and_summarize(resultsPath,outputPath,networks,last_seen_threshold=30,
                           last_seen_min_y=300,threshold=1,rate=1,framerate=50):
    yolo_608 = apply_left_right_bottom_threshold(outputPath + '/yolo_608.csv')
    processingTime = framerate / float(rate)
    summaryDF= None
    for networkname in networks:
        ntwkDF = None
        print(networkname)
        for videoFolder in glob.glob(resultsPath + f'*/*{networkname}'):
            videoName,network = videoFolder.replace(resultsPath,'').split('/')
            print(videoName)
#            detectionDF = pd.read_csv(videoFolder + '/' + videoName + '-' + networkName + '-detection.csv',
#                                      header = [0,1], index_col=0,low_memory=False)
            tp_fn_df = pd.read_csv(videoFolder + '/' + videoName + '-' + networkname + '-detection-TPandFN.csv',
                                   index_col=0,header=0)
            tp_fn_df = tp_fn_df.loc[yolo_608.loc[videoName,:].index,:]
            cols2drop = tp_fn_df.loc[:,'seen2seen':'visible'].columns
            tp_fn_df.drop(labels=cols2drop,axis=1,inplace=True)
            frames = np.ceil(np.arange(0,tp_fn_df.columns.shape[0],processingTime)).astype(np.int)
            frames = frames[frames < tp_fn_df.columns.shape[0]]
            filteredDF = add_metrics_data(cols2drop,tp_fn_df.iloc[:,frames])[cols2drop]
            index = pd.MultiIndex.from_product([[videoName],filteredDF.index.values])
            if ntwkDF is None:
#                cols = pd.MultiIndex.from_product([[networkname],cols2drop])
                ntwkDF = pd.DataFrame(index=index,columns=cols2drop)
            else:
                ntwkDF = ntwkDF.reindex(index=ntwkDF.index.union(index,sort=False))
            ntwkDF.loc[videoName,:] = filteredDF.loc[:,cols2drop].values
#        networkname = ntwk.replace(resultsPath,'').replace('/','').replace('.csv','')
        
        ntwkDF.loc[:,'P_s2s'] = ntwkDF['seen2seen']/(ntwkDF['seen2seen']+ntwkDF['seen2unseen'])
        ntwkDF.loc[:,'P_u2s'] = ntwkDF['unseen2seen']/(ntwkDF['unseen2seen']+ntwkDF['unseen2unseen'])
        ntwkDF.loc[:,'P_visible'] = ntwkDF['seen']/ntwkDF['visible']
        newcols = pd.MultiIndex.from_product([[networkname],ntwkDF.columns.values])
        if summaryDF is None:
            summaryDF = pd.DataFrame(index = ntwkDF.index,
                                     columns = newcols)
            summaryDF[networkname] = ntwkDF
#            return summaryDF,ntwkDF
        else:
#            newDF = pd.DataFrame(index=ntwkDF.index,columns=newcols)
#            newDF.loc[ntwkDF.index,networkname].update(ntwkDF)
            summaryDF = summaryDF.reindex(columns=summaryDF.columns.union(newcols))
#            summaryDF.loc[ntwkDF.index,networkname].update(ntwkDF)
            summaryDF[networkname] = ntwkDF
    return summaryDF
            
    
def summarize_simplified_data(resultsPath,last_seen_threshold=30,last_seen_min_y=300,threshold=1):
    yolo_608 = apply_left_right_bottom_threshold(resultsPath + '/yolo_608.csv')
    summaryDF = None
    for ntwk in glob.glob(resultsPath + '/*.csv'):
        print(ntwk)
        if '608' in ntwk:
            continue
        networkname = ntwk.replace(resultsPath,'').replace('/','').replace('.csv','')
        ntwkDF = pd.read_csv(ntwk,index_col=[0,1],header=[0,1]).loc[yolo_608.index,'metrics_columns']
        ntwkDF.loc[:,'P_s2s'] = ntwkDF['seen2seen']/(ntwkDF['seen2seen']+ntwkDF['seen2unseen'])
        ntwkDF.loc[:,'P_u2s'] = ntwkDF['unseen2seen']/(ntwkDF['unseen2seen']+ntwkDF['unseen2unseen'])
        ntwkDF.loc[:,'P_visible'] = ntwkDF['seen']/ntwkDF['visible']
        newcols = pd.MultiIndex.from_product([[networkname],ntwkDF.columns.values])
        if summaryDF is None:
            summaryDF = pd.DataFrame(index = ntwkDF.index,
                                     columns = newcols)
            summaryDF[networkname] = ntwkDF
#            return summaryDF,ntwkDF
        else:
#            newDF = pd.DataFrame(index=ntwkDF.index,columns=newcols)
#            newDF.loc[ntwkDF.index,networkname].update(ntwkDF)
            summaryDF = summaryDF.reindex(columns=summaryDF.columns.union(newcols))
#            summaryDF.loc[ntwkDF.index,networkname].update(ntwkDF)
            summaryDF[networkname] = ntwkDF
    return summaryDF

def analyse_litters_summary_data(resultPath,summaryPath,visible_threshold=1):
    summaryPath = f'{summaryPath}-ge_threshold-{visible_threshold}'
    print(summaryPath)
    os.makedirs(summaryPath,exist_ok=True)
    for resultCategory in ['TPandFN']:
        networks = ['mobilenetSSD-10000-th0p5-nms0p0-iSz124','mobilenetSSD-10000-th0p5-nms0p0-iSz220','yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128','yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224']
        summaryDF = resample_and_summarize('../data/model_data/','../data/simplified_data',networks)
#        summaryDF = summarize_simplified_data(resultPath)
#        summaryDF = summarise_csv_data_litters(resultPath,resultCategory,visible_threshold)
        if summaryDF is None:
            summaryDF = pd.read_csv(f'{summaryPath}/{resultCategory}_metrics_data.csv',index_col=[0,1],header=[0,1])
#            return summaryDF
#            print(summaryDF.shape)
        else:
            summaryDF.to_csv(f'{summaryPath}/{resultCategory}_metrics_data.csv')
        tex_rows = summaryDF.columns.get_level_values(1).unique()
        tex_df = pd.DataFrame(index=tex_rows)
        s2s_breakdown = pd.DataFrame()
        u2s_breakdown = pd.DataFrame()
        
        for col in summaryDF.columns.get_level_values(0).unique():
            summaryCol = summaryDF[col]
            col = shorten_network_name(col)
            for row in tex_rows:
                if 'P_' in row:
                    tex_df.loc[row,col] = f'${summaryCol[row].mean():.4f} \pm {summaryCol[row].std():.4f}$'
                else:
                    tex_df.loc[row,col] = f'{int(summaryCol[row].sum())}'
            #get extra information
            never_seen = summaryCol.loc[summaryCol['seen'] == 0,'seen'].shape[0]
            tex_df.loc['never_seen',col] = f'{never_seen}'
            
            always_seen = summaryCol.loc[summaryCol['P_visible'] >= 1,'P_visible'].shape[0]
            tex_df.loc['always_seen',col] = f'{always_seen}'
            
            tex_df.loc['nlitter',col] = summaryCol['visible'].count()
            
            #modelling data breakdown
            s2s_breakdown = metrics_data_breakdown(s2s_breakdown,col,'P_s2s',summaryCol)
            u2s_breakdown = metrics_data_breakdown(u2s_breakdown,col,'P_u2s',summaryCol)
            
            #save analysis data
            tex_df.to_latex(f'{summaryPath}/{resultCategory}_metrics_data.tex',escape=False)
            s2s_breakdown.to_latex(f'{summaryPath}/{resultCategory}_s2s_breakdown.tex',escape=False)
            u2s_breakdown.to_latex(f'{summaryPath}/{resultCategory}_u2s_breakdown.tex',escape=False)
        return tex_df,s2s_breakdown,u2s_breakdown
#        summary_csv_data_to_latex(f'{summaryPath}/{resultCategory}_metrics_data')
def generate_plot_for_visibleData(summary_analysis):
    for f in glob.glob(summary_analysis + '*_visibleData.csv'):
        fig = plt.figure()
        ax = fig.gca()
        df = pd.read_csv(f,index_col=0)
#        df['Frequency'].plot()
        df['% of litter'] = df['Frequency'] / df['Frequency'].sum() * 100
        
        df['Cummulative Sum'] = df['% of litter'].cumsum()
        df['Cummulative Sum'].plot(kind='line',style='o',legend=False,ax=ax)
        df['% of litter'].plot(kind='line',style='*',legend=False,ax=ax,secondary_y=True)
        
        plt.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0]],['Cummulative Sum', '% of litter'],loc='center right')
        ax.set_ylabel('Cummulative Sum of %',fontweight='bold')
        ax.right_ax.set_ylabel('% of litter',fontweight='bold')
        ax.set_xlabel('Number of frames',fontweight='bold')
        
        ax.set_xlim(left=df.index.min())
        ax.grid(True)
#        plt.legend(loc='center right')
#        plt.ylabel('Cummulative Percentage')
        fig.savefig(f.replace('.csv','.png'),bbox_inches='tight')

        
if __name__ == '__main__':
    resultPath = '../data/model_data/'
    videosPath = '../data/mp4/'
    summary_analysis = '../data/summary_analysis/'
    fname = '../data/model_data/20190111GOPR9027/mobilenetSSD-10000-th0p5-nms0p0-iSz220/20190111GOPR9027-mobilenetSSD-10000-th0p5-nms0p0-iSz220-detection-TPandFN.csv'
#    s2s_duration,uns2uns_duration = process_consecutive_occurrences(resultPath)
#    s2s_duration = update_duration_columns(s2s_duration)
#    uns2uns_duration = update_duration_columns(uns2uns_duration)
#    process_s2s_uns2uns_data(resultPath,summary_analysis)
#    firstAppearanceDF = first_appearance(resultPath)
#    dataDict = bbox_heatdata(resultPath,summary_analysis,centre=False)
    
#    TPandFN = summarise_csv_data(resultPath,resultCategory = 'TPandFN')
    
#    TPandFN.to_csv(summary_analysis + 'TPandFN.csv')
    experiments = '../data/summary_analysis'
    comp_model = '../data/computational_simulation/comp-sim-100k'
#    compare_fine_grain(experiments,comp_model,'../data/summary_analysis')
#    generate_plot_for_visibleData(experiments + '/')
    