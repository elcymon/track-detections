import cv2 as cv
import numpy as np
import ntpath
import pandas as pd

class TrackDetections:
    def __init__(self,videoName,framesTxt,direction,opticFlow=False,txtType='GT'):
        if opticFlow:
            self.scale = 1
        else:
            self.scale = 0.5
        if txtType == 'GT':
            self.txtHeaders = ['class','x1','y1','x2','y2']
        else:
            self.txtHeaders = ['class','confidence','x1','y1','x2','y2']
        
        self.framesTxtPattern = framesTxt + '/' + ntpath.basename(videoName)[:-4] + '-{:05d}.txt'
        self.videoName = videoName
        self.direction = direction
        self.video = cv.VideoCapture(self.videoName)
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        self.totalFrames = self.video.get(cv.CAP_PROP_FRAME_COUNT) - 1 # 0 based indexing
        

    def readFrame(self,totalFrames,frameNo = None):
        if self.direction == 'reverse' and frameNo is not None:
            self.video.set(cv.CAP_PROP_POS_FRAMES, frameNo)
        
        return self.video.read()
    
    def drawBoxes(self,boxesDF,frame):
        for box in boxesDF.index:
            x1,y1,x2,y2 = boxesDF.loc[box,:]
            cv.rectangle(img=frame,pt1=(x1,y1),pt2=(x2,y2),
                            color=(100,100,100),thickness=1)
        return frame

    def processFrame(self):
        framesNumbers = np.linspace(0,self.totalFrames,self.totalFrames + 1,dtype=np.int)
        if self.direction == 'reverse':
            framesNumbers = np.flip(framesNumbers)
        for frameNo in framesNumbers:
            status,frame = self.readFrame(self.totalFrames,frameNo)
            littersDF = pd.read_csv(self.framesTxtPattern.format(frameNo + 1),
                                    sep=' ',names=self.txtHeaders)
            # print(status,frameNo)
            
            if status:
                if self.scale == 0.5:
                    frame = cv.resize(frame,None,
                                fx=self.scale,fy=self.scale,
                                interpolation=cv.INTER_LANCZOS4)
                # print(littersDF.loc[:,'x1':'y2'])
                frame = self.drawBoxes(littersDF.loc[:,'x1':'y2'],frame)
                cv.imshow('Frame',frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

if __name__ == '__main__':
    dataPath = '../videos/litter-recording/GOPR9027'
    framesTxt = dataPath + '/GOPR9027-yolov3-608/1r1c'
    videoName = dataPath + '/20190111GOPR9027.MP4'
    direction='forward'
    opticFlow=False

    trackDetections = TrackDetections(videoName=videoName,
                                framesTxt=framesTxt,
                                direction=direction,
                                opticFlow=opticFlow)
    trackDetections.processFrame()
