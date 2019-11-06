import cv2 as cv
import numpy as np
import ntpath
import pandas as pd
import iou
import utils
import sys
from track_detections import TrackDetections
from copy import deepcopy
from zipfile import ZipFile

class DetectionMetric:
    def __init__(self,GTcsvFile,videoName,framesTxt,resultPath,\
                    zipFile,opticFlow=True,txtType='detection',\
                    resultVideo = None, imshow= False):
        
        self.opticFlow = opticFlow
        self.framesTxt = framesTxt
        self.imshow = imshow
        self.resultFilePrefix = resultPath
        self.zip = ZipFile(zipFile)
        self.framesTxtPattern = framesTxt + '-{:05d}.txt'
        # self.framesTxtPattern = framesTxt + '/' + ntpath.basename(videoName)[:-4] + '-{:05d}.txt'
        
        self.trackFP = TrackDetections(videoName = videoName,\
                resultPath=resultPath,zipFile=zipFile,\
                framesTxt = framesTxt, direction = 'forward',opticFlow=True,\
                txtType=txtType,resultVideo=resultVideo,imshow=imshow)
        self.framesTxtPattern = self.trackFP.framesTxtPattern
        self.txtHeaders = self.trackFP.txtHeaders

        self.GTDF = TrackDetections(videoName = videoName,\
                resultPath=resultPath,zipFile=zipFile,\
                framesTxt = '', direction = 'forward',opticFlow=True,\
                txtType = 'GT',resultVideo = None,imshow=imshow)
        
        self.GTDF.trackerDF = self.GTDF.readCSVTrackerDF(GTcsvFile)

        #dataframes for storing metrics information for analysis
        self.data2analyse = ['seen2seen','seen2unseen','unseen2seen','unseen2unseen','seen','unseen','visible']
        self.TPandFN = pd.DataFrame(columns=self.data2analyse)
        self.FPdata = pd.DataFrame(columns=self.data2analyse)

        
        # self.trackFP.trackerDF = self.trackFP.readCSVTrackerDF(GTcsvFile)
    def tagBoxes(self,boxes,tag):
        for b in range(len(boxes)):
            if tag == 'FP':
                boxes[b][4] += ',' + tag
            else:
                boxes[b][4] = tag
        return boxes
    def updateMetricsData(self,detectionData,frameNo):
        
        for data in detectionData:
            box,boxID,delx,dely,boxType,interDur = data
            
            if  'L' in boxID:
                df = self.TPandFN
            elif 'F' in boxID:
                df = self.FPdata
            else:
                print(boxID,'is an unknow ID category')
                return
        
            if boxID not in df.index:
                #insert default data for boxID not previously seen
                df.loc[boxID,self.data2analyse] = [0,0,0,0,0,0,1]
            else:
                df.loc[boxID,'visible'] += 1
            
            if 'TP' in boxType or 'iou' in boxType or 'new' in boxType:
                df.loc[boxID,frameNo + 1] = 1
                df.loc[boxID,'seen'] += 1
                if frameNo in df.columns:
                    if df.loc[boxID,frameNo] == 1:
                        df.loc[boxID,'seen2seen'] += 1
                    elif df.loc[boxID,frameNo] == 0:
                        df.loc[boxID,'unseen2seen'] += 1
            elif 'FN' in boxType or 'inter' in boxType:
                df.loc[boxID,frameNo + 1] = 0
                df.loc[boxID,'unseen'] += 1
                if frameNo in df.columns:
                    if df.loc[boxID,frameNo] == 1:
                        df.loc[boxID,'seen2unseen'] += 1
                    elif df.loc[boxID,frameNo] == 0:
                        df.loc[boxID,'unseen2unseen'] += 1
            else:
                print(boxType,'is an unknown box type category')

    def saveMetricsData(self):
        self.TPandFN.to_csv(self.trackFP.resultFilePrefix + '-TPandFN.csv')
        self.FPdata.to_csv(self.trackFP.resultFilePrefix + '-FPdata.csv')

    def saveTrackFP(self):
        self.trackFP.trackerDF.to_csv(self.trackFP.resultFilePrefix + '.csv')

    def processFrame(self):
        prevFP = []
        prevFrameGray = None

        for frameNo in self.trackFP.framesNumbers:
            print(frameNo)
            status, frame = self.trackFP.readFrame(self.trackFP.totalFrames)
            if status:
                frame = self.trackFP.resize(frame)
                if prevFrameGray is None:
                    prevFrameGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                
                frameGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                
                
                #setup trackFP
                if self.trackFP.ellipse_mask is None and \
                    self.trackFP.horizon is not None:

                    _,self.trackFP.ellipse_mask = \
                        self.trackFP.create_ellipse_mask(self.trackFP.horizon,frame)
                if self.trackFP.ellipse_mask is not None:
                    frame = self.trackFP.apply_horizon_image_filter(frame,self.trackFP.ellipse_mask)
                
                #read ground truth data for current frame
                gtdfLitters = self.GTDF.getBoxesFromTrackerDF(frameNo + 1)

                # read network detection for current frame
                littersDF = pd.read_csv(self.zip.open(self.framesTxtPattern.format(frameNo + 1)),
                                    sep=' ', names=self.txtHeaders)
                #filter based on mask
                littersDF = self.trackFP.apply_horizon_points_filter(littersDF, dataType = 'DataFrame')
                frame = self.trackFP.drawBoxes(littersDF.loc[:,'x1':'y2'],frame)
                
                #evaluate iou to filter TP, FN and FP data
                #this will be between gt and littersDF
                if gtdfLitters is None:
                    gtdfLitters = []
                truePositives,falseNegatives,falsePositives = \
                    iou.evaluateIOUs(gtdfLitters,littersDF.loc[:,'x1':'y2'].values,\
                        IOUThreshold=sys.float_info.min)
                
                #track False Positives data
                if len(prevFP) > 0:
                    #eliminate previous False Positives that are outside horizon
                    prevFP = self.trackFP.apply_horizon_points_filter(prevFP)
                    #reduce False Positives that are eventually True Positives in Ground Truth
                    # in current frame
                    # l = len(prevFP)
                    # _,prevFP,_ = \
                    #     iou.evaluateIOUs(prevFP,truePositives,trackType='iou+inter',\
                    #         IOUThreshold=sys.float_info.min)
                    # if l != len(prevFP):
                    #     print('removed {}'.format(l - len(prevFP)))

                fpTracked,fpMissing,fpNew = \
                    iou.evaluateIOUs(prevFP,falsePositives,trackType='iou+inter',\
                        IOUThreshold=sys.float_info.min)
                
                # interpolate missingBoxes
                interpolatedfpMissing = self.trackFP.interpolateNewPosition(\
                                            fpMissing,frameGray,prevFrameGray, centred = True)
                
                # search within missing new FP if interpolated data matches any
                fpTracked2,interpolatedfpMissing,fpNew = \
                    iou.evaluateIOUs(interpolatedfpMissing,fpNew,trackType='iou+inter',\
                        IOUThreshold=sys.float_info.min)
                
                #assign FP ids after filtering all matches from previous frame
                fpNew = self.trackFP.allocateIDs(fpNew, prefix='F')

                falsePositives = fpTracked + fpTracked2 + interpolatedfpMissing + fpNew
                prevFP = deepcopy(falsePositives)
                prevFrameGray = frameGray
                
                #tag detection data
                truePositives = self.tagBoxes(truePositives,'TP')
                falseNegatives = self.tagBoxes(falseNegatives,'FN')
                falsePositives = self.tagBoxes(falsePositives,'FP')

                

                #update FP tracker DF
                detectionData = self.trackFP.updateTrackerDF(frameNo,\
                    truePositives+falseNegatives+falsePositives)
                
                self.updateMetricsData(detectionData,frameNo)

                
                
                #visualize result
                if detectionData is not None:
                    frame = self.trackFP.drawTrackerDF(detectionData,frame,gtDF = False)
                if self.imshow:
                    cv.imshow("Detection Metrics", frame)

                if self.trackFP.vid_writer is not None:
                    self.trackFP.vid_writer.write(frame.astype(np.uint8))
                
                key = cv.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break

if __name__ == '__main__':
    networkName = sys.argv[1]
    videoFile = sys.argv[2]
    imshow = eval(sys.argv[3])

    dataPath = '../data'
    resultPath = dataPath + '/model_data/' + videoFile + '/' + networkName + '/' + videoFile + '-' + networkName
    zipFile = dataPath + '/' + networkName + '.zip'
    framesTxt = networkName + '/1r1c/' + videoFile
    videoName = dataPath + '/mp4/' + videoFile + '.MP4'
    gtNetwork = 'yolov3-litter_10000-th0p0-nms0p0-iSz608'
    GTcsvFile = dataPath + '/model_data/' + videoFile + '/' + gtNetwork + '/' + videoFile + '-' + gtNetwork + '-GT-pruned.csv'
    resultVideo = 'detection'

    detectionMetric = DetectionMetric(GTcsvFile=GTcsvFile,framesTxt=framesTxt,
                zipFile=zipFile,resultPath=resultPath,videoName=videoName,opticFlow=True,
                txtType=resultVideo,resultVideo=resultVideo,imshow=imshow)
    
    detectionMetric.processFrame()

    detectionMetric.saveTrackFP()
    detectionMetric.saveMetricsData()