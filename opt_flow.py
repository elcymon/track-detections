#!/usr/bin/env python

'''
example to show optical flow
USAGE: opt_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv



def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

def main():
    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    cam = cv.VideoCapture(fn)
    ret = False
    while not ret:
        ret, prev = cam.read()
        prev = cv.resize(prev,None,
                fx=0.5,fy=0.5,
                interpolation=cv.INTER_LANCZOS4)
        # print(ret)
    
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()
    
    i = 0
    while True:
        ret, img = cam.read()
        img = cv.resize(img,None,
                                fx=0.5,fy=0.5,
                                interpolation=cv.INTER_LANCZOS4)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        # print(flow[:,:,0])
        # break
        cv.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)

        ch = cv.waitKey(5)
        
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
        print(i)
        i += 1

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()