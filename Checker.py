# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:55:57 2020

@author: RTodo
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


ds=12
lower = np.array([0,0,195])
upper = np.array([26,70,255])

def FeatureExtraction(img_rgb,contour_filter,straightContour):
    img_feature=img_rgb.copy()
    for i,contour in enumerate(contour_filter):
        #center of an object
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # set values as what you need in the situation
            cx, cy = 0, 0
        img = cv.circle(img_feature,(cx,cy), 3, (255,0,0), -1)
                
        #Match Score
        match_score = cv.matchShapes(contour,straightContour,cv.CONTOURS_MATCH_I3,0)
        #Based on match_score
        if match_score < 0.075:
            img = cv.putText(img_feature ,"Good Part",(cx-50,cy+35),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA)         
        else:
            img = cv.putText(img_feature ,"Defect",(cx-50,cy+35),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA) 

    
    # Display extracted features
    plt.figure(figsize = (ds,ds))
    plt.imshow(img_feature)
    plt.axis('off')
    plt.show()    

    return 0



def FilterContours(img_rgb,contours,hierarchy):
    contour_filter=[]
    img_contour=img_rgb.copy()
    for i,contour in enumerate(contours):
        #Remove inside contours
        if hierarchy[0][i][3]==-1:
            #Remove small and large contours
            if cv.contourArea(contour) > 1500 and cv.contourArea(contour)<15000:
                cv.drawContours(img_contour, [contour], -1, (0,0,255), 2)  
                contour_filter.append(contour)
    return contour_filter


def straightShape(imgStraight):
    ksize=5
    img_rgb = cv.cvtColor(imgStraight, cv.COLOR_BGR2RGB)
    img_blur = cv.GaussianBlur(img_rgb,(ksize,ksize),0)
    img_hsv = cv.cvtColor(img_blur, cv.COLOR_RGB2HSV)

    img_pro = cv.inRange(img_hsv, lower, upper)
    contours,hierarchy = cv.findContours(img_pro,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        if hierarchy[0][i][3]==-1:
            return contour


#Main Function
def Check(isStraight, isLeft, isRight):
    imgStraight = cv.imread("straight_shape_2.png")
    straightContour = straightShape(imgStraight)
    for i in range (5):
        ksize=5
        img=cv.imread("./images/straight/group3/opencv_frame_"+str(i+10)+".png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_blur = cv.GaussianBlur(img_rgb,(ksize,ksize),0)
        img_grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_grey_blur = cv.GaussianBlur(img_grey,(ksize,ksize),0)
        img_hsv = cv.cvtColor(img_blur, cv.COLOR_RGB2HSV)
        img_pro = cv.inRange(img_hsv, lower, upper)
        ret,img_otsu = cv.threshold(img_grey_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) 
        if isStraight:
            contours,hierarchy = cv.findContours(img_pro,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours,hierarchy = cv.findContours(img_otsu,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour_filter = FilterContours(img_blur, contours, hierarchy)
        x = FeatureExtraction(img_blur,contour_filter,straightContour)
        
        
Check(1,0,0)