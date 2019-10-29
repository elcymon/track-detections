# Python3 implementation of the approach 
from math import sqrt 
import numpy as np

class BoxData:
    def __init__(self,x1,y1,x2,y2,detID,delx,dely,info,interDur):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.detID = detID
        self.delx = delx
        self.dely = dely
        self.info = info
        self.interDur = interDur
    def getBound(self):
        return self.x1,self.x2,self.y1,self.y2
    
    def getAllBoxData(self):
        return self.x1,self.x2,self.y1,self.y2,self.detID,\
            self.delx,self.dely,self.info,self.interDur

# Function to find the circle on 
# which the given three points lie 
def findCircle(x1, y1, x2, y2, x3, y3) :
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2 
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2 
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (((sx13) * (x12) + (sy13) *
        (x12) + (sx21) * (x13) +
        (sy21) * (x13)) // (2 *
        ((y31) * (x12) - (y21) * (x13))))
            
    g = (((sx13) * (y12) + (sy13) * (y12) +
        (sx21) * (y13) + (sy21) * (y13)) //
        (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
        2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0 
    # where centre is (h = -g, k = -f) and 
    # radius r as r^2 = h^2 + k^2 - c 
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c 

    # r is the radius 
    r = round(sqrt(sqr_of_r), 5)
    pt1_angle = getPointAngle((x1,y1),(h,k))
    pt2_angle = getPointAngle((x3,y3),(h,k))

    return (round(h),round(k)),round(r),pt1_angle,pt2_angle

def getPointAngle(pt,centre):
    return 180 * np.arctan2(pt[1] - centre[1], pt[0] - centre[0]) / np.pi
# Driver code 
if __name__ == "__main__" : 

    x1 = 1 ; y1 = 1
    x2 = 2 ; y2 = 4
    x3 = 5 ; y3 = 3

    print(findCircle(x1, y1, x2, y2, x3, y3))
    horizonPoints = [162,18,59,494,143,937]
    print(findCircle(*horizonPoints))

# This code is contributed by Ryuga 
