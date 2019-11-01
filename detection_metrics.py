import cv2 as cv
import numpy as np
import ntpath
import pandas as pd
import iou
import utils
import sys
from track_detections import TrackDetections
from copy import deepcopy

class DetectionMetric:
    def __init__(self,GTcsvFile,videoName,framesTxt,\
                    opticFlow=True,txtType='detection',\
                    resultVideo = None):
        
        self.opticFlow = opticFlow
        
        self.framesTxtPattern = framesTxt + '/' + ntpath.basename(videoName)[:-4] + '-{:05d}.txt'
        
        self.trackFP = TrackDetections(videoName = videoName,\
                framesTxt = framesTxt, direction = 'forward',opticFlow=True,\
                txtType=txtType,resultVideo=resultVideo)
        self.framesTxtPattern = self.trackFP.framesTxtPattern
        self.txtHeaders = self.trackFP.txtHeaders

        self.GTDF = TrackDetections(videoName = videoName,\
                framesTxt = '', direction = 'forward',opticFlow=True,\
                txtType = 'GT',resultVideo = None)
        
        self.GTDF.trackerDF = self.GTDF.readCSVTrackerDF(GTcsvFile)

        
        # self.trackFP.trackerDF = self.trackFP.readCSVTrackerDF(GTcsvFile)
    def tagBoxes(self,boxes,tag):
        for b in range(len(boxes)):
            boxes[b][4] = tag
        return boxes

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
                littersDF = pd.read_csv(self.framesTxtPattern.format(frameNo + 1),
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
                    l = len(prevFP)
                    _,prevFP,_ = \
                        iou.evaluateIOUs(prevFP,truePositives,trackType='iou+inter',\
                            IOUThreshold=sys.float_info.min)
                    if l != len(prevFP):
                        print('removed {}'.format(l - len(prevFP)))

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
                

                
                
                #visualize result
                if detectionData is not None:
                    frame = self.trackFP.drawTrackerDF(detectionData,frame,gtDF = False)
                
                cv.imshow("Detection Metrics", frame)

                if self.trackFP.vid_writer is not None:
                    self.trackFP.vid_writer.write(frame.astype(np.uint8))
                
                key = cv.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break

if __name__ == '__main__':
    dataPath = '../videos/litter-recording/GOPR9027'
    videoName = dataPath + '/20190111GOPR9027.MP4'
    framesTxt = dataPath + '/GOPR9027-mobilenet-220/1r1c'
    GTcsvFile = dataPath + '/GOPR9027-yolov3-608/1r1c-GT-pruned.csv'
    resultVideo = 'detection'

    detectionMetric = DetectionMetric(GTcsvFile=GTcsvFile,
                videoName=videoName,framesTxt=framesTxt,opticFlow=True,
                txtType=resultVideo,resultVideo=None)
    
    detectionMetric.processFrame()