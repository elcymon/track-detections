import cv2 as cv
import numpy as np
import ntpath
import pandas as pd
import iou
import utils

class TrackDetections:
    def __init__(self,videoName,framesTxt,direction,\
                    horizonPoints = [18,162,494,59,937,143], #[162,18,59,494,143,937],
                    opticFlow=False,txtType='GT'):
        if opticFlow:
            self.scale = 1
        else:
            self.scale = 0.5
        if txtType == 'GT':
            self.txtHeaders = ['class','x1','y1','x2','y2']
        else:
            self.txtHeaders = ['class','confidence','x1','y1','x2','y2']
        
        #frame dimension
        self.xMax = 960
        self.yMax = 540

        self.missingBoxThreshold = 5 #  np.inf #

        self.framesTxtPattern = framesTxt + '/' + ntpath.basename(videoName)[:-4] + '-{:05d}.txt'
        self.videoName = videoName
        self.direction = direction
        self.video = cv.VideoCapture(self.videoName)
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        self.totalFrames = self.video.get(cv.CAP_PROP_FRAME_COUNT) - 1 # 0 based indexing
        self.trackerDF = pd.DataFrame()
        self.horizon = utils.findCircle(*horizonPoints)
        self.ellipse_mask = None
        self.vid_writer = cv.VideoWriter(self.videoName[:-4] + '-tracking.avi',
                                    cv.CAP_FFMPEG, cv.VideoWriter_fourcc(*'X264'),
                                    50,(self.xMax,self.yMax))
        

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

    def allocateIDs(self,newBoxes):
        litterID = len(self.trackerDF.columns)
        for i in range(len(newBoxes)):
            newBoxes[i][1] = 'L{}'.format(litterID)
            
            litterID += 1
        return newBoxes
    
    def enforceBounds(self,data,maxValue):
        for i in range(len(data)):
            if data[i] >= maxValue:
                data[i] = int(round(maxValue - 1))
        return data
                    
    def interpolateNewPosition(self,missingBoxes,frameGray,prevFrameGray):
        if frameGray is None or prevFrameGray is None:
            return missingBoxes
        flow = cv.calcOpticalFlowFarneback(prevFrameGray, frameGray, None,\
                                         0.5, 3, 15, 3, 5, 1.2, 0)
        for i in range(len(missingBoxes)):
            oldx1,oldy1,oldx2,oldy2 = missingBoxes[i][0]
            # print(oldx1,oldy1,oldx2,oldy2)
            oldx1,oldx2 = self.enforceBounds([oldx1,oldx2],self.xMax)
            oldy1,oldy2 = self.enforceBounds([oldy1,oldy2],self.yMax)
            # print(oldx1,oldy1,oldx2,oldy2)
            
            flowx1,flowy1 = flow[oldy1,oldx1]
            flowx2,flowy2 = flow[oldy2,oldx2]
            delx = (flowx1 + flowx2) / 2.0
            dely = (flowy1 + flowy2) / 2.0
            newBox = (oldx1+delx,oldy1+dely,\
                        oldx2+delx,oldy2+dely)
            # print(newBox)
            newBox = tuple(int(round(i)) for i in newBox)

            #rearrange points
            # newx1,newy1,newx2,newy2 = newBox
            # newBox = tuple([np.min([newx1,newx2]),np.min([newy1,newy2]),\
            #                 np.max([newx1,newx2]),np.max([newy1,newy2])])
            # print(flowx1,flowy1,flowx2,flowy2,newBox)
            if newBox[0] >= self.xMax or newBox[1] >= self.yMax\
                or newBox[2] <= 0 or newBox[3] <= 0 or missingBoxes[i][5] > self.missingBoxThreshold:
                #out of maximum frame
                missingBoxes[i] = None

            else:#still within frame
                missingBoxes[i][0] = newBox
                missingBoxes[i][2] = round(delx,1)
                missingBoxes[i][3] = round(dely,1)
                missingBoxes[i][4] = 'inter'
        return missingBoxes
    
    def updateTrackerDF(self,frameNo,allBoxes):
        for data in allBoxes:
            if data is not None:
                self.trackerDF.loc[frameNo + 1,data[1]] = str(data)
    
    
    def drawTrackerDF(self,trackerRow,frame):
        for data in trackerRow:
            if data is not None:
                box,boxID,delx,dely,boxType,interDur = data
                x1,y1,x2,y2 = box
                if boxType == 'iou':
                    color = (255,0,0)
                elif boxType == 'inter':
                    color = (0,0,255)
                elif boxType == 'new':
                    color = (255,0,255)
                else:
                    print('invalid boxType: ',boxType)
                # print(box)
                cv.rectangle(img=frame,pt1=(int(x1),int(y1)),
                                pt2=(int(x2),int(y2)),
                                color=color,thickness=2)
        return frame
    def maskFromBoxes(self,boxes,frame):
        mask = np.zeros_like(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        
        for b in boxes:
            
            x1,y1,x2,y2 = b[0]
            mask[y1:y2+1,x1:x2+1] = 1
        
        return mask
    def create_ellipse_mask(self,horizon,frame):
        center,radius,start_angle,end_angle = horizon
        start_angle = 0; end_angle = 360

        axes = (radius,radius)
        angle = 0
        color = (255,255,255)
        thickness = -1
        lineType = cv.LINE_AA
        shift = 0
        ellipse_mask = np.zeros_like(cv.cvtColor(frame,cv.COLOR_BGR2GRAY))

        ellipse_mask = cv.ellipse(ellipse_mask,center,axes,angle,start_angle,end_angle,color,thickness,lineType,shift)
        return cv.threshold(ellipse_mask,1,255,cv.THRESH_BINARY)

    def apply_horizon_image_filter(self,frame):
        
        horizon_ellipse = cv.bitwise_and(frame,frame,mask = self.ellipse_mask)
            # cv.imshow('horizon_ellipse', horizon_ellipse)
        return horizon_ellipse

    def apply_horizon_points_filter(self,boxes):
        filtered_points = []
        for b in boxes:
            x1,y1,x2,y2 = b[0]
            #invert x and y to that of opencv shape/format
            yc,xc = round((x1 + x2) / 2), round((y1 + y2) / 2)
            yc = self.enforceBounds([yc],self.xMax)[0]
            xc = self.enforceBounds([xc],self.yMax)[0]
            
            
            if self.ellipse_mask[int(xc),int(yc)] > 0:
                filtered_points.append(b)
            # print(b,self.ellipse_mask[int(xc),int(yc)])
        
        return filtered_points

    def processFrame(self):
        framesNumbers = np.linspace(0,self.totalFrames,self.totalFrames + 1,dtype=np.int)
        if self.direction == 'reverse':
            framesNumbers = np.flip(framesNumbers)
        prevlittersDF = []
        prevFrameGray = None

        for frameNo in framesNumbers:
            littersDF = pd.read_csv(self.framesTxtPattern.format(frameNo + 1),
                                    sep=' ',names=self.txtHeaders)
            # print(status,frameNo)
            trackedBoxes,missingBoxes,newBoxes =\
                iou.evaluatIOUs(prevlittersDF,littersDF.loc[:,'x1':'y2'].values,IOUThreshold=0.001)
            
            # assign new IDs for newBoxes
            newBoxes = self.allocateIDs(newBoxes)

            # interpolate missingBoxes
            status,frame = self.readFrame(self.totalFrames,frameNo)
            frameGray = None

            if status:
                
                if self.scale == 0.5:
                    frame = cv.resize(frame,None,
                                fx=self.scale,fy=self.scale,
                                interpolation=cv.INTER_LANCZOS4)
                # mask = self.maskFromBoxes(trackedBoxes,frame)
                if self.ellipse_mask is None:
                    _,self.ellipse_mask = self.create_ellipse_mask(self.horizon,frame)
                # frame = self.apply_horizon_image_filter(frame)
                # trackedBoxes = self.apply_horizon_points_filter(trackedBoxes)
                # missingBoxes = self.apply_horizon_points_filter(missingBoxes)
                # newBoxes = self.apply_horizon_points_filter(newBoxes)

                maskedFrame = frame#cv.bitwise_and(frame,frame,mask=mask)
                if prevFrameGray is None:
                    
                    prevFrameGray = cv.cvtColor(maskedFrame,cv.COLOR_BGR2GRAY)
                
                frameGray = cv.cvtColor(maskedFrame,cv.COLOR_BGR2GRAY)
                
            interpolatedMissingBoxes = self.interpolateNewPosition(missingBoxes,frameGray,prevFrameGray)
            
            prevFrameGray = frameGray
            
            # insert all into the tracker dataframe
            self.updateTrackerDF(frameNo,trackedBoxes + newBoxes + interpolatedMissingBoxes)
            # if frameNo == 166:
            #     print(self.trackerDF.loc[frameNo + 1, :].values)
            prevlittersDF = []
            
            if frameNo + 1 in self.trackerDF.index:
                for i in self.trackerDF.loc[frameNo + 1, :]:
                    if i is not None:
                        try:
                            prevlittersDF.append(eval(i))
                        except TypeError:
                            pass
                            # print(i)           
                    else:
                        print("i is none")

            # print(self.trackerDF)
            print(frameNo + 1, len(trackedBoxes), len(newBoxes),len(interpolatedMissingBoxes))
            
            
            if status:
                
                    # mask = cv.resize(mask,None,fx=self.scale,fy=self.scale,interpolation=cv.INTER_LANCZOS4)
                # print(littersDF.loc[:,'x1':'y2'])
                # mask = self.maskFromBoxes(trackedBoxes,frame)
                # mask = cv.bitwise_and(frame,frame,mask=mask)
                frame = self.drawBoxes(littersDF.loc[:,'x1':'y2'],frame)
                frame = self.drawTrackerDF(prevlittersDF,frame)
                
                
                # cv.imshow('Ellipse Mask',self.ellipse_mask)
                cv.imshow('Frame',frame)
                self.vid_writer.write(frame.astype(np.uint8))
                # cv.imshow("mask",mask)
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
    # trackerDFTxt = framesTxt[::-1].replace('/','trackerDF-',1)[::-1]
    # print('saving to: %s' % trackerDFTxt)
    trackDetections.trackerDF.to_csv(framesTxt + '-trackerDF.csv')
