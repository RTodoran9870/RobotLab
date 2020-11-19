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
minArea = 6282.0
minPerimeter = 453.44
ksize=3
itern=2
kernel = np.ones((ksize,ksize),np.uint8)

def FeatureExtraction(img_rgb,contour_filter,straightContour,defect1Contour,defect2Contour):
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
        #Area and perimeter
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour,True)
        #Match Score
        match_score_list = []
        tags = ["Good Part", "Defect: Head Cut Off","Defect: Head Cut Off + Filled","Cut in half"]
        match_score = cv.matchShapes(contour,straightContour,cv.CONTOURS_MATCH_I2,0)
        match_score_list.append(match_score)
        defect_1_match_score = cv.matchShapes(contour,defect1Contour, cv.CONTOURS_MATCH_I2,0)
        match_score_list.append(defect_1_match_score)
        defect_2_match_score = cv.matchShapes(contour,defect2Contour, cv.CONTOURS_MATCH_I2,0)
        match_score_list.append(defect_2_match_score)
        if area < 0.70 * minArea and perimeter < 0.70 * minPerimeter:
            match_score_list.append(0.00)
        else:
            match_score_list.append(1.00)
        img = cv.putText(img_feature,tags[np.argmin(match_score_list)],(cx-50,cy+35),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA)
    
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
    img_grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    img_grey_blur = cv.GaussianBlur(img_grey,(ksize,ksize),0)
    ret,img_otsu = cv.threshold(img_grey_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) 
    contours,hierarchy = cv.findContours(img_otsu,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        if hierarchy[0][i][3]==-1:
            return contour


#Main Function
def Check(isStraight, isLeft, isRight):
    imgStraight = cv.imread("./shapes/straight_shape.png")
    imgDefect1 = cv.imread("./shapes/straight_defect_1.png")
    imgDefect2 = cv.imread("./shapes/straight_defect_2.png")
    straightContour = straightShape(imgStraight)
    defect1Contour =  straightShape(imgDefect1)
    defect2Contour = straightShape(imgDefect2)
    for i in range (8):
        ksize=5
        img=cv.imread("./images/straight/group3/opencv_frame_"+str(i*4)+".png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_grey_blur = cv.GaussianBlur(img_grey,(ksize,ksize),0)
        ret,img_otsu = cv.threshold(img_grey_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        img_pro=cv.morphologyEx(img_otsu, cv.MORPH_OPEN, kernel)
        if isStraight:
            contours,hierarchy = cv.findContours(img_pro,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours,hierarchy = cv.findContours(img_pro,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour_filter = FilterContours(img_rgb, contours, hierarchy)
        x = FeatureExtraction(img_rgb,contour_filter,straightContour,defect1Contour,defect2Contour)
        
        
Check(1,0,0)