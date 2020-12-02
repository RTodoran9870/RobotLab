# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:09:10 2020

@author: RTodo
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from save_results import ResultsSave
import Checker
#from gpio import LED
#from gpio import Button


def videoCapture():
    cap = cv.VideoCapture("video4.avi")
    i=10

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.rectangle(frame,(560,120),(640,200),(255,0,0),5)
        frame_cropped = frame[120:200,560:640]

        if i%1==0:
            avg_color_per_row = np.average(frame_cropped, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            #print(avg_color)
            if avg_color[0]<90:
                i=0
                print(avg_color)
            if i==1:
                Checker.testCameraCurved(frame)
                partList, Pass=Checker.cropCurvedImage(frame, True)
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
        i+=1

# When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    
    
videoCapture()