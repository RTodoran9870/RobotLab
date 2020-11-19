# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:55:57 2020
@author: RTodo
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


ds=12
minStraightArea = 6282.0
minStraightPerimeter = 453.44
minCurvedArea = 9101.0
minCurvedPerimeter = 551.65
ksize=3
itern=2
kernel = np.ones((ksize,ksize),np.uint8)




def FeatureExtraction(img_rgb,contour_filter,modelContour,defectContourList,isStraight):
    img_feature=img_rgb.copy()
    
    #Initialize local Counters and lists to hold coord. of good parts and defects
    goodPartCounter = 0
    defectCounter = 0

    part_cx_list=[]
    part_cy_list=[]
    defects_cx_list=[]
    defects_cy_list=[]

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
        
        #Area perimeter and aspect ratio
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour,True)
        x,y,w,h = cv.boundingRect(contour)
        aspect_ratio = round(float(w)/h, 2)
        
        #Match Score
        match_score_list = []
        if isStraight:
            tags = ["Train","Good Part", "Defect: Head Cut Off","Defect: Head Cut Off + Filled","Defect: Filled in","Defect: Cut in half","Hole inside"]
            if perimeter<400 and perimeter>200 and area<6000 and area>4000 and aspect_ratio<0.6 and aspect_ratio>0.35:
                match_score_list.append(0.00)
            else:
                match_score_list.append(1.00)
        else:
            tags = ["Good Part", "Defect: Filled in","Defect: Cut in half","Hole Inside"]
        
        if isStraight:
            match_score = cv.matchShapes(contour,modelContour,cv.CONTOURS_MATCH_I3,0)
        else:
            match_score = cv.matchShapes(contour,modelContour,cv.CONTOURS_MATCH_I1,0)            
        match_score_list.append(match_score)
        
        for j in defectContourList:
            if isStraight:
                defect_match_score = cv.matchShapes(contour,j,cv.CONTOURS_MATCH_I3,0)
            else:
                defect_match_score = cv.matchShapes(contour,j,cv.CONTOURS_MATCH_I1,0)                
            match_score_list.append(defect_match_score)
        if isStraight:
            if area < 0.70 * minStraightArea and perimeter < 0.70 * minStraightPerimeter:
                match_score_list.append(0.01)
            else:
                match_score_list.append(1.00)
        else:
            if area < 0.70 * minCurvedArea and perimeter < 0.70 * minCurvedPerimeter:
                match_score_list.append(0.01)
            else:
                match_score_list.append(1.00)
        
        #Check if there is a hole inside
        if area<500:
            tag = "Defect: Hole inside"
        else:    
            tag = tags[np.argmin(match_score_list)]
        
        if tag == "Good Part": 
            part_cx_list.append(cx)
            part_cy_list.append(cy)
            goodPartCounter += 1
            
        else:
            defects_cx_list.append(cx)
            defects_cy_list.append(cy)
            defectCounter += 1
        
        
        img = cv.putText(img_feature,tag,(cx-50,cy+35),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA)
        
    
    # Display extracted features
    plt.figure(figsize = (ds,ds))
    plt.imshow(img_feature)
    plt.axis('off')
    plt.show()    

    return part_cx_list,part_cy_list,defects_cx_list,defects_cy_list,goodPartCounter,defectCounter



def FilterContours(img_rgb,contours,hierarchy):
    contour_filter=[]
    img_contour=img_rgb.copy()
    for i,contour in enumerate(contours):
        #Remove inside contours
        if hierarchy[0][i][2]==-1:
            #Remove small and large contours
            if cv.contourArea(contour) > 1500 and cv.contourArea(contour)<15000:
                cv.drawContours(img_contour, [contour], -1, (0,0,255), 2)  
                contour_filter.append(contour)
            if hierarchy[0][i][3]>=15:
                cv.drawContours(img_contour, [contour], -1, (0,0,255), 2)  
                contour_filter.append(contour)
    return contour_filter


def getShape(imgStraight):
    ksize=5
    img_rgb = cv.cvtColor(imgStraight, cv.COLOR_BGR2RGB)
    img_grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    img_grey_blur = cv.GaussianBlur(img_grey,(ksize,ksize),0)
    ret,img_otsu = cv.threshold(img_grey_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) 
    contours,hierarchy = cv.findContours(img_otsu,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    plt.figure(figsize = (ds,ds))
    plt.imshow(img_otsu)
    plt.axis('off')
    plt.show()  
    for i,contour in enumerate(contours):
        if hierarchy[0][i][3]==-1:
            return contour


#Main Function
def Check(isStraight, isLeft, isRight):
    defectContourList = []
    if isStraight:
        imgStraight = cv.imread("./shapes/straight_shape.png")
        imgDefect1 = cv.imread("./shapes/straight_defect_1.png")
        imgDefect2 = cv.imread("./shapes/straight_defect_2.png")
        imgDefect3 = cv.imread("./shapes/straight_defect_3.png")
        straightContour = getShape(imgStraight)
        defect1Contour =  getShape(imgDefect1)
        defect2Contour = getShape(imgDefect2)
        defect3Contour = getShape(imgDefect3)
        defectContourList.append(defect1Contour)
        defectContourList.append(defect2Contour)
        defectContourList.append(defect3Contour)
    if isLeft or isRight:
        imgCurved = cv.imread("./shapes/curved_shape.png")
        imgDefect1 = cv.imread("./shapes/curved_defect_1.png")
        
        curvedContour = getShape(imgCurved)
        defect1Contour =  getShape(imgDefect1)
        
        defectContourList.append(defect1Contour)
        
    for i in range (5):
        ksize=5
        img=cv.imread("./images/straight/group5_defects/opencv_frame_"+str(28+i*7)+".png")
        if isRight:
            img=cv.flip(img, 0)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_grey_blur = cv.GaussianBlur(img_grey,(ksize,ksize),0)
        ret,img_otsu = cv.threshold(img_grey_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        img_pro=cv.morphologyEx(img_otsu, cv.MORPH_OPEN, kernel)
        plt.figure(figsize = (ds,ds))
        plt.imshow(img_pro)
        plt.axis('off')
        plt.show() 
        contours,hierarchy = cv.findContours(img_pro,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour_filter = FilterContours(img_rgb, contours, hierarchy)
        if isStraight:
            part_cx_list,part_cy_list,defects_cx_list,defects_cy_list,goodPartCounter,defectCounter = FeatureExtraction(img_rgb,contour_filter,straightContour,defectContourList,isStraight)
        else:
            part_cx_list,part_cy_list,defects_cx_list,defects_cy_list,goodPartCounter,defectCounter = FeatureExtraction(img_rgb,contour_filter,curvedContour,defectContourList,isStraight)
        if goodPartCounter == 12:
            print("Pass")
        elif goodPartCounter + defectCounter == 12:
            print("Fail")
        elif goodPartCounter + defectCounter < 12:
            print('Missing Parts' + '\n')
            print("Fail")
        elif goodPartCounter + defectCounter < 13:
            # Gather all non empty positions
            nonEmpty_cx_list=part_cx_list+defects_cx_list
            nonEmpty_cy_list=part_cy_list+defects_cy_list

            # Remove the train position to get the parts grid
            j = nonEmpty_cx_list.index(min(nonEmpty_cx_list))
            cy2 = nonEmpty_cy_list[j]
            nonEmpty_cx_list.pop(j)
            nonEmpty_cy_list.pop(j)

            #get coordinates of possible positions
            cx1 = min(nonEmpty_cx_list)
            cx2 = (max(nonEmpty_cx_list) - min(nonEmpty_cx_list))/2
            cx3 = max(nonEmpty_cx_list)
            cy1 = min(nonEmpty_cy_list)
            #cy2 = min(nonEmpty_cy_list) + (max(nonEmpty_cy_list) - min(nonEmpty_cy_list))/3
            cy3 = min(nonEmpty_cy_list) + 2*(max(nonEmpty_cy_list) - min(nonEmpty_cy_list))/3
            cy4 = max(nonEmpty_cx_list)

            num_emptySlots = 13 - (goodPartCounter + defectCounter)

            #for emptySlot in range(num_emptySlots):



            print(str(num_emptySlots) + ' missing Parts')
            print("Fail" + '\n')
            
        print("Good Parts:" + str(goodPartCounter),"Defects:" + str(defectCounter))
        print(part_cx_list,part_cy_list,defects_cx_list,defects_cy_list)
    
            
            
        
        #Reset conters and lists to hold coord. of good parts and defects
        goodPartCounter = 0
        defectCounter = 0

        part_cx_list=[]
        part_cy_list=[]
        defects_cx_list=[]
        defects_cy_list=[]
        
Check(1,0,0)