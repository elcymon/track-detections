import os
import sys
from collections import Counter
import numpy as np
import pandas as pd
from copy import deepcopy
import utils

def getCentre(box):
    return ((box[0] + box[2]) / 2.0,(box[1] + box[3]) / 2.0)

def computeCenterShiftXY(boxA,boxB):
    centreA = getCentre(boxA)
    centreB = getCentre(boxB)
    return (round(centreB[0] - centreA[0],2), round(centreB[1] - centreA[1]),2)

def evaluateIOUs(gts,dects,trackType='iou',IOUThreshold=sys.float_info.min):
    #each gts element is [(x1,y1,x2,y2),id,delx,dely,info,interDur]
    # delx,dely are gradient in x,y direction, which is None initially
    # info could be new, iou or inter.
    # interDur is the number of frames this object has been missing, default is 0

    #each dects value is (x1,y1,x2,y2)
    trackedBox,missingBox,newBox = [],[],[]

    for d in range(len(dects)):
        iouMax = sys.float_info.min
        jMax = None
        #transform to desired format
        if trackType == 'iou':

            detectionData = [list(dects[d]),None,None,None,None,0]
        else:
            detectionData = dects[d]
        
        #find maximum overlap of this box with gts data
        for j in range(len(gts)):
            # print(gts)
            # print(detectionData)
            # print(detectionData,gts)
            iou = computeIOU(detectionData[0], gts[j][0])
            # if gts[j][1] == 'L102':
            #     print(iou,detectionData[0], gts[j])
            if iou > iouMax:
                iouMax = iou
                jMax = j
        
        #if overlap is above threshold, it is tracked, else, this is a new box
        if iouMax >= IOUThreshold and jMax is not None:
            detectionData[1] = gts[jMax][1]
            gradientXY = computeCenterShiftXY(gts[jMax][0],detectionData[0])
            detectionData[2] = gradientXY[0]
            detectionData[3] = gradientXY[1]
            # tType = 'iou'
            # if trackType == 'iou+inter':
            #     tType = 'iou'

            detectionData[4] = 'iou' # iou tracked box
            trackedBox.append(detectionData)
            if len(gts) > 0:
                del gts[jMax]
        else:
            detectionData[4]  = 'new'
            newBox.append(detectionData)
    if trackType == 'iou':
        for b in gts:
            if b[4] == 'inter':
                b[5] += 1
            missingBox.append(b)
    else:
        missingBox = gts
    
    return trackedBox,missingBox,newBox

def computeIOU(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    if iou < 0:
        print(boxA,boxB,interArea,union,iou)
        return 0
    assert iou >= 0
    return iou

def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)


def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


