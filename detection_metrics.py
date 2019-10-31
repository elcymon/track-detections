import cv2 as cv
import numpy as np
import ntpath
import pandas as pd
import iou
import utils
import sys
from track_detections import TrackDetections

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

        self.trackFP.trackerDF = self.trackFP.readCSVTrackerDF(GTcsvFile)
    
    def processFrame(self):
        for frameNo in self.trackFP.framesNumbers:
            print(frameNo)
            status, frame = self.trackFP.readFrame(self.trackFP.totalFrames)
            if status:
                frame = self.trackFP.resize(frame)
                
                #setup trackFP
                if self.trackFP.ellipse_mask is None and \
                    self.trackFP.horizon is not None:

                    _,self.trackFP.ellipse_mask = \
                        self.trackFP.create_ellipse_mask(self.trackFP.horizon,frame)
                if self.trackFP.ellipse_mask is not None:
                    frame = self.trackFP.apply_horizon_image_filter(frame,self.trackFP.ellipse_mask)
                
                #read ground truth data for current frame
                gtdfLitters = self.trackFP.getBoxesFromTrackerDF(frameNo + 1)

                # read network detection for current frame
                littersDF = pd.read_csv(self.framesTxtPattern.format(frameNo + 1),
                                    sep=' ', names=self.txtHeaders)
                frame = self.trackFP.drawBoxes(littersDF.loc[:,'x1':'y2'],frame)
                
                #
                
                
                #visualize result
                if gtdfLitters is not None:
                    frame = self.trackFP.drawTrackerDF(gtdfLitters,frame,gtDF = True)
                
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