# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ntpath
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import utils

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

def process_consecutive_occurrences(resultsPath,resultCategory):
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
        
        metrics_data = get_detection_metric_data(filename)
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
                        
def fine_grain_s2s_uns2uns_data(summary_analysis=None,computational_simulation=None,minT=1,maxT=20):
    #fine grain analysis/histogram of s2s and uns2uns data to visualize their histograms
    for csvPattern in ['*s2s_duration.csv', '*uns2uns_duration.csv']:
        latexDF = None
        for csvData in glob.glob(summary_analysis + csvPattern):
            if 'binned' in csvData:#skip binned data
                continue
            csvDataDF = pd.read_csv(csvData,header=[0,1],index_col = 0, low_memory = False)
            csvDataDF = csvDataDF.loc[(minT <= csvDataDF.index) & (csvDataDF.index <= maxT),:]
            
            csvDataDF.drop(labels=['FreqxSteps','PctFreqxSteps'],axis=1,level=1,inplace=True)
            for ntwk in csvDataDF.columns.get_level_values(0).unique():
                print(ntwk)
                ntwkShort = shorten_network_name(ntwk)
                if latexDF is None:
                    latexDF = pd.DataFrame(columns = [ntwkShort], index = csvDataDF.index)
                else:
                    latexDF = latexDF.reindex(columns = latexDF.columns.union([ntwkShort]))
                latexDF.loc[csvDataDF.index,ntwkShort] = \
                    ['{:.0f} ({:.2f})'.format(d,pctd) for d,pctd in zip(csvDataDF[ntwk]['Frequency'],csvDataDF[ntwk]['PctFreq'])]
            
            #output filename
            fname = csvData.replace('.csv',f'_range{minT}-{maxT}')
            #save as latex table
            latexDF.to_latex(fname + '.tex',escape=False)
            #plot data
            #rename column names
            pctFreqDF = csvDataDF.xs('PctFreq',level=1,drop_level=False,axis=1)
            pctFreqDF.columns = [shorten_network_name(i) for i in pctFreqDF.columns.droplevel(1)]
            plot_metric_data(pctFreqDF,fname + '.png',ylim=[0,100])
#            csvDataDF  = csvDataDF.xs(['Frequency','PctFreq'],axis=1,drop_level=False,level=1)
#            print(csvDataDF.index)
#            return latexDF
        
    
           
            
def process_s2s_uns2uns_data(resultsPath,summary_analysis):
    for resultCategory in ['TPandFN','FPdata']:
        print(resultCategory)
        s2s_duration,uns2uns_duration = process_consecutive_occurrences(resultsPath,resultCategory)
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
    
    ntwk += networkName.replace('iSz','').split('-')[-1]
    return ntwk

def plot_metric_data(df,filename,legend=True,ylim=[0,1]):
    f = plt.figure()
    ax = f.gca()
    df.plot(kind='line',style='-o',legend=False,ax = ax,ylim=ylim)
#    print(np.arange(0,df.shape[0],1,dtype=np.int))
    ax.set_xticks(np.arange(1,df.shape[0] + 1,1,dtype=np.int))
    plt.legend(loc='best',ncol=2)
    f.savefig(filename,bbox_inches='tight')
    
def summary_csv_data_to_latex(filename):
    df = pd.read_csv(filename + '.csv',header=[0,1],index_col=0)
    metricSUM = ['visible','seen','unseen','seen2seen','seen2unseen','unseen2seen','unseen2unseen']
    metricMean = ['P_visible','P_s2s','P_u2s']
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
    unique_litterDF = df['yolov3-litter_10000-th0p0-nms0p0-iSz608']
#    print(unique_litterDF.columns)
    unique_litterDF.columns = [i.replace('nlitter_','') for i in unique_litterDF.columns]
    if 'TPandFN' in filename:
        plot_metric_data(unique_litterDF,filename + 'nlitter.png',ylim=[0,unique_litterDF.max().max() * 1.1])
        print(unique_litterDF.sum())
        
    for metric in ['P_s2s','P_u2s','P_visible']:
        metricDF = df.xs(metric,level=1,drop_level=False,axis=1)
        metricDF.columns = [shorten_network_name(i) for i in metricDF.columns.droplevel(1)]
        plot_metric_data(metricDF,filename + '-' + metric + '.png')
    
    return df
def generate_extra_summary_columns(df):
    df2 = pd.DataFrame()
    for row in df.index:
        rowData = df.loc[row,:].dropna()
        df2.loc[row,'firstState'] = rowData[0]
        
        seenData = rowData[rowData == 1]#information about indexes where litter is seen
        if len(seenData) > 0:
            firstSeenIdx = seenData.index[0]
            
            df2.loc[row,'unseen_uns2s'] = 1
            totalUnseen_uns = len(rowData[:firstSeenIdx]) - 1
            df2.loc[row,'unseen_uns2uns'] = totalUnseen_uns - 1 if totalUnseen_uns > 0 else 0 #minus the time-step it was seen
            
            #process uns2uns and uns2s for duration that litter has been seen
            seenData = rowData[firstSeenIdx:]
#            print(seenData[0])
#            print(firstSeenIdx,consecutive_occurrences(seenData,0).sum(),rowData)
            totalUnseen = consecutive_occurrences(seenData,0)
            
            df2.loc[row,'seen_uns2uns'] = totalUnseen.sum() - len(totalUnseen) if len(totalUnseen) > 0 else 0
            df2.loc[row,'seen_uns2s'] = sum(seenData.diff().dropna().astype(np.int) == 1)
        else:
            df2.loc[row,'unseen_uns2s'] = 0
            df2.loc[row,'unseen_uns2uns'] = len(rowData) - 1 if len(rowData) > 0 else 0#was seen the whole time
     
    return df2

def summarise_csv_data(resultsPath,resultCategory):
    
    summaryDF = pd.DataFrame()
    summaryCols = ['seen2seen','seen2unseen','P_s2s',\
                'unseen2seen','unseen2unseen','P_u2s',\
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
            rawDataDF = pd.read_csv(csvFile,index_col=0,header=0)
            
            #logged information from raw data
            csvDF = rawDataDF.loc[:,'seen2seen':'visible']
            
            #from raw data extract state (i.e. seen/unseen) for first occurrence in GT, 
            #seen_uns2s, seen_uns2uns, unseen_uns2s, unseen_uns2uns
            #DATA
            #return csvDF,rawDataDF
            extraDataDF = generate_extra_summary_columns(rawDataDF.drop(labels=csvDF.columns,axis=1))
            csvDF = pd.concat([extraDataDF,csvDF])
            summaryCols = list(extraDataDF.columns.values) + summaryCols
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
            summaryDF.loc[videoName,dataHeader] = csvDFsummary.loc[summaryCols].values
#            summaryDF.merge(csvDF.sum(),how='left',on=videoName,left_on=networkName)
#            csvDF = csvDF
#            return csvDF
    return summaryDF

def close2frame_border_mask(fwidth,fheight,edgewidth,horizon):
    center,radius,start_angle,end_angle = utils.findCircle(*horizon)
    maskDF = pd.DataFrame(index=np.arange(1,fheight+1,1,dtype=np.int),
                          columns=np.arange(1,fwidth+1,1,dtype=np.int),dtype=np.bool)
    maskDF.loc[:,:] = False
    
    for x in maskDF.columns:
#        theta = np.arccos((x - center[0])/float(radius))
        radiusY = center[1] -  np.sqrt(np.square(radius) - np.square(x - center[0]))  #radius * np.sin(theta) - fheight#
        print(x)#,radiusY,theta)
        if False and (x <= edgewidth or x >= maskDF.columns[-1] - edgewidth):
            ymax = maskDF.index[-1]
        else:
            ymax = np.rint(radiusY + edgewidth)#/2.0)
        ymin = maskDF.index[0]#np.rint(radiusY)# - edgewidth/2.0)
        
        maskDF.loc[ymin:ymax,x] = True
    
#    sns.heatmap(maskDF.astype(np.int),vmax=1)
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

def first_appearance(resultsPath):
    firstAppearanceDF = pd.DataFrame(index=np.arange(1,541,1,dtype=np.int),
                                     columns=np.arange(1,961,1,dtype=np.int))
    firstAppearanceDF.loc[:,:] = 0
    for videoFolder in glob.glob(resultsPath + '/*/*608*'):
        videoName,networkName = videoFolder.replace(resultsPath,'').split('/')
        csvDF = pd.read_csv(videoFolder + '/' + videoName + '-' + networkName + '-GT-pruned.csv',
                            header = [0,1], index_col = 0, low_memory = False)
        litterIDs = csvDF.columns.get_level_values(0).unique()
        for lit in litterIDs:
            firstOccurrence = csvDF[lit].dropna(axis=0,how='all').head(1)
            cx = firstOccurrence[['x1','x2']].mean(axis=1).astype(int).values[0]
            cy = firstOccurrence[['y1','y2']].mean(axis=1).astype(int).values[0]
#            print(cx,cy)
            
            firstAppearanceDF.loc[cy,cx] += 1
#            print(firstOccurrence)
            
        print(videoFolder)
    sns.heatmap(firstAppearanceDF,annot=False,xticklabels=[],yticklabels=[])
    sns.heatmap(firstAppearanceDF,annot=False,xticklabels=[],yticklabels=[],vmax=1)
    return firstAppearanceDF
def create_frameDim_DF():
    frame = pd.DataFrame(index=np.arange(1,541,1,dtype=np.int),
                                     columns=np.arange(1,961,1,dtype=np.int))
    frame.loc[:,:] = 0
    return frame
def heatmap(df,figName,vmax,colorbar=True):
    f = plt.figure(figsize=(16,9))
    sns.heatmap(df,annot=False,xticklabels=[],yticklabels=[],ax=f.gca(),vmax=vmax,cbar=colorbar)
    f.savefig(figName, bbox_inches='tight')
    
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
    
    TPandFN = summarise_csv_data(resultPath,resultCategory = 'TPandFN')
    
#    TPandFN.to_csv(summary_analysis + 'TPandFN.csv')
    