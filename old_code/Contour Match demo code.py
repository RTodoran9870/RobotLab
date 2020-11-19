# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:19:17 2020

@author: RTodo
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


#Getting the images
img1=cv2.imread("./images/straight/group5_correct/opencv_frame_0.png")
img2=cv2.imread("image3.jpg")
img3=cv2.imread("image4.jpg")


#Transforming them into grey images
img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img_hsv1 = cv2.cvtColor(img_rgb1, cv2.COLOR_RGB2HSV)
img_grey1 = cv2.cvtColor(img_rgb1, cv2.COLOR_RGB2GRAY)

img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_hsv2 = cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2HSV)
img_grey2 = cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2GRAY)

img_rgb3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img_hsv3 = cv2.cvtColor(img_rgb3, cv2.COLOR_RGB2HSV)
img_grey3 = cv2.cvtColor(img_rgb3, cv2.COLOR_RGB2GRAY)

#Gaussian Blur followed by OTSU
ksize=5

img_pro1 = cv2.GaussianBlur(img_grey1,(ksize,ksize),0)
img_pro2 = cv2.GaussianBlur(img_grey2,(ksize,ksize),0)
img_pro3 = cv2.GaussianBlur(img_grey3,(ksize,ksize),0)

ret1,img_otsu1 = cv2.threshold(img_pro1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2,img_otsu2 = cv2.threshold(img_pro2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3,img_otsu3 = cv2.threshold(img_pro3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img_pro1=img_otsu1.copy()
img_pro2=img_otsu2.copy()
img_pro3=img_otsu3.copy()

#Getting the contours
ksize=3
itern=2
kernel = np.ones((ksize,ksize),np.uint8)

contours1,hierarchy1 = cv2.findContours(img_pro1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2,hierarchy2 = cv2.findContours(img_pro2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours3,hierarchy3 = cv2.findContours(img_pro3,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt1 = contours1[1]
cnt2 = contours1[2]
cnt3 = contours2[0]

#Matching the contours
ret12 = cv2.matchShapes(cnt1,cnt2,1,0.0)
ret13 = cv2.matchShapes(cnt1,cnt3,1,0.0)

#Prinitng the match scores (the lower the better)
print(ret12)
print(ret13)

#contours,hierarchy = cv2.findContours(img1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnt1 = contours[0]
#contours,hierarchy = cv2.findContours(img2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnt2 = contours[0]
#ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
#print( ret )