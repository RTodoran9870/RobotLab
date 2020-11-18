# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:50:59 2020

@author: Alexander
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Grey colour
"""
def changeColor(img):
    img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb1, cv2.COLOR_RGB2HSV)
    img_grey = cv2.cvtColor(img_rgb1, cv2.COLOR_RGB2GRAY)
    return img_grey 
"""


#Getting the images
img1=cv.imread("image5.png")


#Transforming them into grey images
img_rgb1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img_grey1 = cv.cvtColor(img_rgb1, cv.COLOR_RGB2GRAY)

#Debugging
plt.imshow(img_grey1,'gray')
plt.axis('off')
plt.show()

#Gaussian Blur 
ksize=5
img_pro1 = cv.GaussianBlur(img_grey1,(ksize,ksize),0)

#OTSU Thresholding
ret1,img_otsu1 = cv.threshold(img_pro1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
img_pro1=img_otsu1.copy()

#Debugging
plt.imshow(img_pro1,'gray')
plt.axis('off')
plt.show()

#Getting the contours
contours1,hierarchy1 = cv.findContours(img_pro1,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#Debugging
print('hierarchy',hierarchy1)

#Filter contours
contour_filter=[]
img_contour=img_rgb1.copy()
for i,contour in enumerate(contours1):
    if hierarchy1[0][i][3]==-1:
        cv.drawContours(img_contour, [contour], -1, (0,0,255), 2)  
        contour_filter.append(contour)
        
    if hierarchy1[0][i][2]==-1:
        cv.drawContours(img_contour, [contour], -1, (0,0,255), 2)  
        contour_filter.append(contour)
        
 
# Display contours   
ds=12        
plt.figure(figsize = (ds,ds))
plt.imshow(img_contour)
plt.axis('off')
plt.show()        

# Feature Extraction
area_list=[]
aspect_ratio_list=[]
perimeter_list=[]

img_feature=img_rgb1.copy()
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

    #The area of an object
    area = cv.contourArea(contour)
    img = cv.putText(img_feature ,"area: "+str(int(area)),(cx-50,cy+25),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA) 

    #The perimeter of an object
    perimeter = cv.arcLength(contour,True)
    img = cv.putText(img_feature ,"perimeter: "+str(int(perimeter)),(cx-50,cy+15),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA) 
    img = cv.putText(img_feature ,"number: "+str(i),(cx-50,cy+5),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA) 

    area_list.append(area)
    aspect_ratio_list.append(aspect_ratio)
    perimeter_list.append(perimeter)
    
# Display extracted features
plt.figure(figsize = (ds,ds))
plt.imshow(img_feature)
plt.axis('off')
plt.show()    

# Display features for sanity check
i = 0
for area in area_list:
    print(i+1,area_list[i],aspect_ratio_list[i],perimeter_list[i])
    i+=1
