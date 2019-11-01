import cv2 as cv
import numpy as np
import ntpath
import pandas as pd
import iou
import utils
import sys

class TrackDetections:
    def __init__(self,videoName,framesTxt,direction,\
                    horizonPoints = [18,162,494,59,937,143], #[162,18,59,494,143,937],
                    opticFlow=False,txtType='GT',resultVideo = None):
        self.boxData = ['x1','y1','x2','y2','id','delx','dely','info','interDur']

        self.opticFlow = opticFlow
        self.scale = 0.5

        if txtType == 'GT':
            self.txtHeaders = ['class','x1','y1','x2','y2']
        else:
            self.txtHeaders = ['class','confidence','x1','y1','x2','y2']
        
        self.litterID = 0 # id to be used by next litter

        #frame dimension
        self.xMax = 960
        self.yMax = 540

        self.missingBoxThreshold = np.inf #5 #  

        self.framesTxtPattern = framesTxt + '/' + ntpath.basename(videoName)[:-4] + '-{:05d}.txt'
        self.videoName = videoName
        self.direction = direction
        self.video = cv.VideoCapture(self.videoName)
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        self.totalFrames = self.video.get(cv.CAP_PROP_FRAME_COUNT) - 1 # 0 based indexing
        
        self.framesNumbers = np.linspace(0,self.totalFrames,self.totalFrames + 1,dtype=np.int)
        if self.direction == 'reverse':
            self.framesNumbers = np.flip(self.framesNumbers)

        self.trackerDF = pd.DataFrame()
        if horizonPoints is not None:
            self.horizon = utils.findCircle(*horizonPoints)
        else:
            self.horizon = None
        
        self.ellipse_mask = None
        if resultVideo is not None:
            self.vid_writer = cv.VideoWriter(self.videoName[:-4] + '-' + resultVideo + '.avi',
                                    cv.CAP_FFMPEG, cv.VideoWriter_fourcc(*'X264'),
                                    50,(self.xMax,self.yMax))
        else:
            self.vid_writer = None
        

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

    def allocateIDs(self,newBoxes,prefix = 'L'):
        # prefix could be L or FP, i.e. litter or False Positive
        
        
        for i in range(len(newBoxes)):
            newBoxes[i][1] = '{}{}'.format(prefix,self.litterID)
            # print(litterID)
            
            self.litterID += 1
        return newBoxes
    
    def enforceBounds(self,data,maxValue):
        for i in range(len(data)):
            if data[i] >= maxValue:
                data[i] = int(round(maxValue - 1))
        return data
                    
    def interpolateNewPosition(self,missingBoxes,frameGray,prevFrameGray,centred=False):
        if frameGray is not None and prevFrameGray is not None:
            flow = cv.calcOpticalFlowFarneback(prevFrameGray, frameGray, None,\
                                         0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            flow = None
        interpolationResult = []
        for i in missingBoxes:
            oldx1,oldy1,oldx2,oldy2 = i[0]
            if self.opticFlow or flow is None:
                # print(oldx1,oldy1,oldx2,oldy2)
                oldx1,oldx2 = self.enforceBounds([oldx1,oldx2],self.xMax)
                oldy1,oldy2 = self.enforceBounds([oldy1,oldy2],self.yMax)
                # print(oldx1,oldy1,oldx2,oldy2)
                
                flowx1,flowy1 = flow[oldy1,oldx1]
                flowx2,flowy2 = flow[oldy2,oldx2]
                
                delx = (flowx1 + flowx2) / 2.0
                dely = (flowy1 + flowy2) / 2.0
                if centred:
                    newBox = (oldx1+delx,oldy1+dely,\
                                oldx2+delx,oldy2+dely)
                else:
                    newBox = (oldx1+flowx1,oldy1+flowy1,\
                                oldx2+flowx2,oldy2+flowy2)
                # print(newBox)
                newBox = tuple(int(round(i)) for i in newBox)
            else:
                delx = 0
                dely = 0
                newBox = (oldx1,oldy1,oldx2,oldy2)

            #rearrange points
            # newx1,newy1,newx2,newy2 = newBox
            # newBox = tuple([np.min([newx1,newx2]),np.min([newy1,newy2]),\
            #                 np.max([newx1,newx2]),np.max([newy1,newy2])])
            # print(flowx1,flowy1,flowx2,flowy2,newBox)
            if not (newBox[0] >= self.xMax or newBox[1] >= self.yMax\
                or newBox[2] <= 0 or newBox[3] <= 0 or i[5] > self.missingBoxThreshold):
                i[0] = newBox
                i[2] = round(delx,1)
                i[3] = round(dely,1)
                i[4] = 'inter'
                interpolationResult.append(i)
        return interpolationResult
    
    def updateTrackerDF(self,frameNo,allBoxes):
        currDetections = []
        for data in allBoxes:
            if data is not None:
                boxID = data[1]
                dataHeader = pd.MultiIndex.from_product([boxID,self.boxData],names=['boxID','boxData'])
                # print(dataHeader)
                if len(self.trackerDF.columns) == 0:
                    self.trackerDF = pd.DataFrame(columns=dataHeader)
                    # self.trackerDF.columns.names
                else:
                    # print(self.trackerDF.columns.union(dataHeader))
                    self.trackerDF = self.trackerDF.reindex(columns=self.trackerDF.columns.union(dataHeader))
                # print(self.trackerDF)
                self.trackerDF.loc[frameNo + 1, dataHeader] = list(data[0]) + data[1:]
                # for col,value in zip(dataHeader,data[0] + data[1:]):
                #     self.trackerDF.loc[frameNo + 1,tuple(col)] = value
                currDetections.append(data)
        return currDetections
    
    
    def drawTrackerDF(self,trackerRow,frame,gtDF = False):
        for data in trackerRow:
            if data is not None:
                box,boxID,delx,dely,boxType,interDur = data
                x1,y1,x2,y2 = box
                if boxType == 'FN' or gtDF:
                    color = (150,50,100)
                elif boxType == 'TP' or boxType == 'iou':
                    color = (255,0,0)
                elif boxType == 'FP' or boxType == 'inter':
                    color = (0,0,255)
                    if boxType == 'FP':
                        frame = cv.putText(frame, boxID, (int(x2),int(y1)), \
                                cv.FONT_HERSHEY_PLAIN, 1, color, 2, cv.LINE_8, False)
                elif boxType == 'new':
                    color = (255,0,255)
                    frame = cv.putText(frame, boxID, (int(x2),int(y1)), \
                                cv.FONT_HERSHEY_PLAIN, 1, color, 2, cv.LINE_8, False)
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

    def apply_horizon_image_filter(self,frame,horizon_mask):
        
        horizon_ellipse = cv.bitwise_and(frame,frame,mask = horizon_mask)
            # cv.imshow('horizon_ellipse', horizon_ellipse)
        return horizon_ellipse

    def apply_horizon_points_filter(self,boxes,dataType=None):
        if dataType == 'DataFrame':
            filtered_points = pd.DataFrame(columns=boxes.columns)
            boxes = boxes.values
            i = 0
        else:
            filtered_points = []
        for b in boxes:
            # print(b)
            if dataType == 'DataFrame':
                # print(b)
                clss,conf,x1,y1,x2,y2 = b
            else:
                x1,y1,x2,y2 = b[0]
            #invert x and y to that of opencv shape/format
            yc,xc = round((x1 + x2) / 2), round((y1 + y2) / 2)
            yc = self.enforceBounds([yc],self.xMax)[0]
            xc = self.enforceBounds([xc],self.yMax)[0]
            
            
            if self.ellipse_mask[int(xc),int(yc)] > 0:
                if dataType == 'DataFrame':
                    filtered_points.loc[i,:] = b
                    i += 1
                else:
                    filtered_points.append(b)
            # print(b,self.ellipse_mask[int(xc),int(yc)])
        
        return filtered_points
    def pruneTrackerDF(self):
        pruneColumns = []
        for boxID in self.trackerDF.columns.get_level_values(0).unique():
            # print(boxID)
            infoColumn = self.trackerDF.loc[:,(boxID,'info')]
            iouCount = len(infoColumn[infoColumn == 'iou'])
            interpolationCount = len(infoColumn[infoColumn == 'inter'])
            # print(boxID,iouCount,interpolationCount)

            if iouCount == 0:
                pruneColumns.append(boxID)
            elif interpolationCount / float(interpolationCount + iouCount) > 0.7:
                #if proportion of interpolationCount is more than 70%
                pruneColumns.append(boxID)
        if len(pruneColumns) > 0:
            print("dropping ",pruneColumns)
            self.trackerDF.drop(labels=pruneColumns, axis=1, inplace = True)
    def getBoxesFromTrackerDF(self,frameNo):
        boundBoxes = []
        if frameNo in self.trackerDF.index:
            #process frame data and drop na values/columns
            boxesData = pd.DataFrame(self.trackerDF.loc[frameNo,:])
            boxesData = boxesData.T.stack(dropna=False).unstack(0)
            boxesData.columns = boxesData.columns.get_level_values(0)
            boxesData = boxesData.dropna(axis=1,thresh=2).T
            
            for boxID,boxData in boxesData.iterrows():
                
                # print(boxData)
                boxBound = [(int(boxData.x1),int(boxData.y1),int(boxData.x2),int(boxData.y2)),\
                                boxData.id,boxData.delx,boxData.dely,\
                                boxData.info,boxData.interDur]
                
                boundBoxes.append(boxBound)
        if len(boundBoxes) == 0:
            boundBoxes = None
        return boundBoxes
    def readCSVTrackerDF(self,csvFileName):
        return pd.read_csv(csvFileName,header=[0,1],index_col=0,low_memory=False)
    def resize(self,frame):
        return cv.resize(frame,None,fx=self.scale,fy=self.scale,interpolation=cv.INTER_LANCZOS4)
    
    def visualizeTrackerDF(self,csvFileName = None):
        if csvFileName is not None:
            self.trackerDF = self.readCSVTrackerDF(csvFileName)
            print(self.trackerDF.shape)

        self.video = cv.VideoCapture(self.videoName)
        if self.vid_writer is not None: #there is an instance of vid_writer already, record pruned trackerDF
            self.vid_writer = cv.VideoWriter(self.videoName[:-4] + '-pruned.avi',
                                    cv.CAP_FFMPEG, cv.VideoWriter_fourcc(*'X264'),
                                    50,(self.xMax,self.yMax))
        for frameNo in self.framesNumbers:
            print(frameNo)
            status, frame = self.readFrame(self.totalFrames,frameNo)
            if status:
                frame = self.resize(frame)
                
                if self.ellipse_mask is None and self.horizon is not None:
                    #create horizon mask
                    _,self.ellipse_mask = self.create_ellipse_mask(self.horizon,frame)
                if self.ellipse_mask is not None:
                    frame = self.apply_horizon_image_filter(frame,self.ellipse_mask)
                
                littersDF = self.getBoxesFromTrackerDF(frameNo + 1) 
                if littersDF is not None:
                    frame = self.drawTrackerDF(littersDF, frame)
                

                cv.imshow("TrackerDF",frame)
                if self.vid_writer is not None:
                    self.vid_writer.write(frame.astype(np.uint8))

                key = cv.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
    def processFrame(self):
        
        prevlittersDF = []
        prevFrameGray = None

        for frameNo in self.framesNumbers:
            littersDF = pd.read_csv(self.framesTxtPattern.format(frameNo + 1),
                                    sep=' ',names=self.txtHeaders)
            # print(status,frameNo)
            trackedBoxes,missingBoxes,newBoxes =\
                iou.evaluateIOUs(prevlittersDF,littersDF.loc[:,'x1':'y2'].values,IOUThreshold=sys.float_info.min)
            
            
            status,frame = self.readFrame(self.totalFrames,frameNo)
            frameGray = None

            if status:
                
                if self.scale == 0.5:
                    frame = cv.resize(frame,None,
                                fx=self.scale,fy=self.scale,
                                interpolation=cv.INTER_LANCZOS4)
                # mask = self.maskFromBoxes(trackedBoxes,frame)
                if self.ellipse_mask is None and self.horizon is not None:
                    _,self.ellipse_mask = self.create_ellipse_mask(self.horizon,frame)
                
                if self.ellipse_mask is not None:
                    # frame = self.apply_horizon_image_filter(frame,self.ellipse_mask)
                    trackedBoxes = self.apply_horizon_points_filter(trackedBoxes)
                    missingBoxes = self.apply_horizon_points_filter(missingBoxes)
                    newBoxes = self.apply_horizon_points_filter(newBoxes)

                # maskedFrame = frame#cv.bitwise_and(frame,frame,mask=mask)
                if prevFrameGray is None:
                    
                    prevFrameGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                
                frameGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                
            
            # interpolate missingBoxes
            interpolatedMissingBoxes = self.interpolateNewPosition(missingBoxes,frameGray,prevFrameGray)
            if self.opticFlow:
                trackedBoxes2,interpolatedMissingBoxes,newBoxes = \
                    iou.evaluateIOUs(interpolatedMissingBoxes,newBoxes,trackType='iou+inter',\
                        IOUThreshold=sys.float_info.min)
                trackedBoxes = trackedBoxes + trackedBoxes2

            # assign new IDs for newBoxes
            newBoxes = self.allocateIDs(newBoxes)
            # if self.opticFlow:
            #     dects = [b[0] for b in newBoxes]
            #     print(len(interpolatedMissingBoxes),len(dects))
            #     trackedBoxes2,interpolatedMissingBoxes,newBoxes =\
            #         iou.evaluatIOUs(interpolatedMissingBoxes,dects,trackType='inter+iou',IOUThreshold=0.001)
            #     trackedBoxes = trackedBoxes + trackedBoxes2
            #     print(len(trackedBoxes2),len(interpolatedMissingBoxes), len(newBoxes))
                # print(newBoxes)

            prevFrameGray = frameGray
            
            # insert all into the tracker dataframe
            prevlittersDF = \
                        self.updateTrackerDF(frameNo,\
                                    trackedBoxes + newBoxes + interpolatedMissingBoxes)
            # if frameNo == 166:
            #     print(self.trackerDF.loc[frameNo + 1, :].values)
            # prevlittersDF = []
            
            # if frameNo + 1 in self.trackerDF.index:
            #     for i in self.trackerDF.loc[frameNo + 1, :]:#trackedBoxes + newBoxes + interpolatedMissingBoxes
            #         if i is not None:
            #             try:
            #                 prevlittersDF.append(eval(i))
            #             except TypeError:
            #                 # print('TypeError: {}'.format(i))
            #                 pass
            #                 # print(i)           
            #         else:
            #             print("i is None: {}".format(i))

            # print(self.trackerDF)
            print(frameNo + 1, len(trackedBoxes), len(newBoxes),len(interpolatedMissingBoxes))
            
            
            if status:
                
                    # mask = cv.resize(mask,None,fx=self.scale,fy=self.scale,interpolation=cv.INTER_LANCZOS4)
                # print(littersDF.loc[:,'x1':'y2'])
                # mask = self.maskFromBoxes(trackedBoxes,frame)
                # mask = cv.bitwise_and(frame,frame,mask=mask)
                
                
                
                if self.ellipse_mask is not None:
                    frame = self.apply_horizon_image_filter(frame,self.ellipse_mask)

                frame = self.drawBoxes(littersDF.loc[:,'x1':'y2'],frame)
                frame = self.drawTrackerDF(prevlittersDF,frame)
                
                
                # cv.imshow('Ellipse Mask',self.ellipse_mask)
                cv.imshow('Frame',frame)
                if self.vid_writer is not None:
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
    horizonPoints = [18,162,494,59,937,143] # x1,y1, x2,y2, x3,y3
    opticFlow=True
    txtType='GT'
    resultVideo = 'GT'#None # string for name of experiment/type

    trackDetections = TrackDetections(videoName=videoName, resultVideo = resultVideo,
                                framesTxt = framesTxt,
                                direction = direction, horizonPoints = horizonPoints,
                                opticFlow = opticFlow, txtType = txtType)
    if resultVideo is None:
            dfName = framesTxt 
    else:
        dfName = framesTxt + '-' + resultVideo 

    
    trackDetections.processFrame()
    
    # print(trackDetections.trackerDF.columns)
    boxIDsList = trackDetections.trackerDF.columns.get_level_values(0).unique()
    # print(boxIDsList)
    print('total litter {}, last litter: {}'.format(len(boxIDsList),boxIDsList[-1]))
    trackDetections.trackerDF.to_csv(dfName + '-unpruned.csv')

    print('pruning')
    trackDetections.pruneTrackerDF()
    boxIDsList = trackDetections.trackerDF.columns.get_level_values(0).unique()
    print('total litter {}, last litter: {}'.format(len(boxIDsList),boxIDsList[-1]))
    
    # trackerDFTxt = framesTxt[::-1].replace('/','trackerDF-',1)[::-1]
    # print('saving to: %s' % trackerDFTxt)
    
    
    trackDetections.trackerDF.to_csv(dfName + '-pruned.csv')

    trackDetections.visualizeTrackerDF(dfName + '-pruned.csv')

