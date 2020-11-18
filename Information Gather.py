# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:01:46 2020

@author: RTodo
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Used for figures
ds=12

#Used for manual filterring
lower = np.array([0,0,195])
upper = np.array([26,70,255])

"""
FeatureExtraction: Extracts the features (area, perimeter, aspect ratio and match score) for all
contours sent
INPUTS: img_rgb: image, used for plotting on it
contour_filter: list of contours to consider
straightContour: contour of the model straight track for matching
OUTPUTS: area_list: list of contour areas
aspect_ratio_list: list of box aspect ratios
perimeter_list: list of contour perimeters
match_score_list: list of match scores
"""
def FeatureExtraction(img_rgb,contour_filter,straightContour):
    #Initialize lists
    area_list=[]
    aspect_ratio_list=[]
    perimeter_list=[]
    match_score_list=[]

    img_feature=img_rgb.copy()
    
    #For each contour sent
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
                
        #The aspect ratio, which is the width divided by the height of the bounding rectangle
        x,y,w,h = cv.boundingRect(contour)
        aspect_ratio = round(float(w)/h, 2)
        img = cv.rectangle(img_feature,(x,y),(x+w,y+h),(0,255,0),2)
        img = cv.putText(img_feature ,"aspect_ratio: "+str(aspect_ratio),(cx-50,cy+35),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA) 
                
        #Match Score
        match_score = cv.matchShapes(contour,straightContour,cv.CONTOURS_MATCH_I3,0)
        
        #The area of an object
        area = cv.contourArea(contour)
        img = cv.putText(img_feature ,"area: "+str(int(area)),(cx-50,cy+25),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA) 
                
        #The perimeter of an object
        perimeter = cv.arcLength(contour,True)
        img = cv.putText(img_feature ,"perimeter: "+str(int(perimeter)),(cx-50,cy+15),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA) 
        img = cv.putText(img_feature ,"number: "+str(i),(cx-50,cy+5),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA) 

        #Appenidng to lists
        area_list.append(area)
        aspect_ratio_list.append(aspect_ratio)
        perimeter_list.append(perimeter)
        match_score_list.append(match_score)
    
    # Display extracted features
    plt.figure(figsize = (ds,ds))
    plt.imshow(img_feature)
    plt.axis('off')
    plt.show()    

    # Display features for sanity check
    i = 0
    for area in area_list:
        print(i+1,area_list[i],aspect_ratio_list[i],perimeter_list[i],match_score_list[i])
        i+=1
    #Return lists
    return area_list, aspect_ratio_list, perimeter_list, match_score_list



"""
FilterContours: function to filter the required contours from all other contour appearing after the filtering
INPUTS: img_rgb: image used for plotting on it
contours: list of all contours found after filtering
hierarchy: Hierarchy of the found contours
OUTPUTS: contour_filter: list of all the needed contours
"""
def FilterContours(img_rgb,contours,hierarchy):
    contour_filter=[]
    img_contour=img_rgb.copy()
    for i,contour in enumerate(contours):
        #Only the contours not included into any other contours
        if hierarchy[0][i][3]==-1:
            #Eliminate very small and very large areas
            if cv.contourArea(contour) > 1500 and cv.contourArea(contour)<15000:
                #Append contours to output
                cv.drawContours(img_contour, [contour], -1, (0,0,255), 2)  
                contour_filter.append(contour)
    plt.figure(figsize = (ds,ds))
    plt.imshow(img_contour)
    plt.axis('off')
    plt.show() 
    return contour_filter

"""
Function to write the data into a txt file
"""
def writeTXT(area_list, aspect_ratio_list, perimeter_list,match_score_list):
    f = open("straight_tracks_training_data_1","a")
    i=0
    for area in area_list:
        f.write(str(area_list[i])+" "+str(aspect_ratio_list[i])+" "+str(perimeter_list[i])+" "+str(match_score_list[i])+"\n")
        i+=1
    f.close()

"""
Function to get the straight contour from the model
"""
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
    

"""
Main function
"""
def DataGather(isStraight, isLeft, isRight):
    imgStraight = cv.imread("straight_shape_2.png")
    #Get contour of the straight shape model
    straightContour = straightShape(imgStraight)
    #The range number shows how many pictures there are in the folder
    for i in range (22):
        #Blur ksize
        ksize=5
        #Reading and processing the image
        img=cv.imread("./images/straight/group5_correct/opencv_frame_"+str(i)+".png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_blur = cv.GaussianBlur(img_rgb,(ksize,ksize),0)
        img_grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_grey_blur = cv.GaussianBlur(img_grey,(ksize,ksize),0)
        img_hsv = cv.cvtColor(img_blur, cv.COLOR_RGB2HSV)
        img_pro = cv.inRange(img_hsv, lower, upper)
        ret,img_otsu = cv.threshold(img_grey_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) 
        #For the straight pieces of track a manual filtering seems to work better (although should be cheked with pictures from the other teams)
        if isStraight:
            contours,hierarchy = cv.findContours(img_pro,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #For the curved pieces, the otsu filterring seems to work better
        else:
            contours,hierarchy = cv.findContours(img_otsu,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        #Plot the information
        plt.figure(figsize = (ds,ds))
        plt.imshow(img_pro)
        plt.axis('off')
        plt.show() 
        
        #Get the required contours
        contour_filter = FilterContours(img_blur, contours, hierarchy)
        #Get the required data
        area_list, aspect_ratio_list, perimeter_list, match_score_list = FeatureExtraction(img_blur,contour_filter,straightContour)
        #OVerall information (commented since it is not always needed)
        """
        print ("Based on 126 straight tracks")
        print("Max area: ", max(area_list))
        print("Min area: ",min(area_list))
        print("Mean area: ",np.mean(area_list))
        print("Max aspect ratio: ",max(aspect_ratio_list))
        print("Min aspect ratio: ",min(aspect_ratio_list))
        print("Mean aspect ratio: ",np.mean(aspect_ratio_list))
        print("Max perimeter: ",max(perimeter_list))
        print("Min perimeter: ",min(perimeter_list))
        print("Mean perimeter: ",np.mean(perimeter_list))
        print("Max match score: ",max(match_score_list))
        print("Min match score: ",min(match_score_list))
        print("Mean match score: ",np.mean(match_score_list))
        """
        #Write the information to a TXT file (not always needed)
        #writeTXT(area_list,aspect_ratio_list,perimeter_list,match_score_list)
        
        
        
        
        
   
    
    

DataGather(True,False,False)