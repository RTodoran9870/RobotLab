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
from time import sleep
#from gpio import LED
#from gpio import Button

ds=12


def videoCapture():
    cap = cv.VideoCapture("video11.avi")
    flag = 0

    while(flag==0):
    # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.rectangle(frame,(5,160),(20,180),(255,0,0),2)
        frame_cropped = frame[160:180,5:20]


        avg_color_per_row = np.average(frame_cropped, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        #print(avg_color)
        if avg_color[0]<110:
            flag=1
            #print(avg_color)
    # Display the resulting frame
    for i in range(24):
        ret, frame = cap.read()
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_check = cv.rectangle(frame,(595,150),(615,170),(255,0,0),2)
    partList, Pass=Checker.cropCurvedImage(frame, True)
    Checker.testCameraCurved(frame_check)
    plt.figure(figsize = (ds,ds))
    plt.imshow(frame)
    plt.axis('off')
    plt.show()

# When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    
    
videoCapture()