import numpy as np
import cv2
from matplotlib import pyplot as plt
import random 
import math

img_bgr = cv2.imread('image2.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


ds=12

img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

plt.figure(figsize = (ds,ds))
plt.imshow(img_grey,'gray')
plt.axis('off')
plt.show()

img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

plt.figure(figsize = (ds,ds))
plt.imshow(img_hsv)
plt.axis('off')
plt.show()

#14, 102, 202
lower = np.array([0,50,150])
upper = np.array([25,150,250])
img_color = cv2.inRange(img_hsv, lower, upper)

f, axarr = plt.subplots(1,2)
axarr[0].imshow(img_rgb)
axarr[1].imshow(img_color)

ksize=5

img_pro = cv2.GaussianBlur(img_grey,(ksize,ksize),0)

ret,img_global = cv2.threshold(img_pro,100,255,cv2.THRESH_BINARY)
img_adaptive = cv2.adaptiveThreshold(img_pro,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,127, 11)
ret2,img_otsu = cv2.threshold(img_pro,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

h,w=img_grey.shape
img_up = np.concatenate((img_grey, img_global), axis=1)
img_down = np.concatenate((img_adaptive, img_otsu), axis=1)
img_all = np.concatenate((img_up, img_down), axis=0)
plt.figure(figsize = (ds,ds))
plt.imshow(img_all,'gray')
plt.text(w, h, "global=100",fontsize=40)
plt.text(0, h*2, "adaptive",fontsize=40)
plt.text(w, h*2, "otsu",fontsize=40)
plt.axis('off')
plt.show()

img_pro=img_otsu.copy()

ksize=3
itern=2
kernel = np.ones((ksize,ksize),np.uint8)

contours,hierarchy = cv2.findContours(img_pro,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print('hierarchy',hierarchy)

contour_filter=[]
img_contour=img_rgb.copy()
for i,contour in enumerate(contours):
    if hierarchy[0][i][3]==-1:
        cv2.drawContours(img_contour, [contour], -1, (0,0,255), 2)  
        contour_filter.append(contour)
        
        
plt.figure(figsize = (ds,ds))
plt.imshow(img_contour)
plt.axis('off')
plt.show()

area_list=[]
aspect_ratio_list=[]

img_feature=img_rgb.copy()
for i,contour in enumerate(contour_filter):
    #center of an object
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    img = cv2.circle(img_feature,(cx,cy), 3, (255,0,0), -1)
    
    #The aspect ratio, which is the width divided by the height of the bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = round(float(w)/h, 2)
    img = cv2.rectangle(img_feature,(x,y),(x+w,y+h),(0,255,0),2)
    img = cv2.putText(img_feature ,"aspect_ratio: "+str(aspect_ratio),(cx-50,cy+35),cv2.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv2.LINE_AA) 

    #The area of an object
    area = cv2.contourArea(contour)
    img = cv2.putText(img_feature ,"area: "+str(int(area)),(cx-50,cy+25),cv2.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv2.LINE_AA) 

    #The perimeter of an object
    perimeter = cv2.arcLength(contour,True)
    img = cv2.putText(img_feature ,"perimeter: "+str(int(perimeter)),(cx-50,cy+15),cv2.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv2.LINE_AA) 
    img = cv2.putText(img_feature ,"number: "+str(i),(cx-50,cy+5),cv2.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv2.LINE_AA) 

    area_list.append(area)
    aspect_ratio_list.append(aspect_ratio)
    

plt.figure(figsize = (ds,ds))
plt.imshow(img_feature)
plt.axis('off')
plt.show()

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img_grey, None)
imgKp1 = cv2.drawKeypoints(img_grey, kp1, None)
plt.figure(figsize = (ds,ds))
plt.imshow(imgKp1)
plt.axis('off')
plt.show()