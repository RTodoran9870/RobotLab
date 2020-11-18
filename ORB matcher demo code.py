# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:55:07 2020

@author: RTodo
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

ds=12

img1 = cv2.imread("straight_track.png",cv2.COLOR_RGB2GRAY)
img2 = cv2.imread("opencv_frame_1.png",cv2.COLOR_RGB2GRAY)
img3 = cv2.imread("image4.jpg",cv2.COLOR_RGB2GRAY)



orb=cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
kp3, des3 = orb.detectAndCompute(img3, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append([m])
print(len(good))
img_matches2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

matches = bf.knnMatch(des1, des3, k=2)

good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append([m])
print(len(good))
img_matches3 = cv2.drawMatchesKnn(img1, kp1, img3, kp3, good, None, flags=2)



plt.figure(figsize = (ds,ds))
plt.imshow(img_matches2)
plt.axis('off')
plt.show()

plt.figure(figsize = (ds,ds))
plt.imshow(img_matches3)
plt.axis('off')
plt.show()