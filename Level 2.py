# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:55:57 2020
@author: RTodo
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#from gpio import LED
#from gpio import Button


ds=12
#Precalibrating ethalons
minStraightArea = 6282.0
minStraightPerimeter = 453.44
minCurvedArea = 8879.5
minCurvedPerimeter = 541.65
ksize=3
itern=2
kernel = np.ones((ksize,ksize),np.uint8)

#Cropping settings
cropping_cy_straight=[35,130,225,320]
cropping_cy_length_straight=95
cropping_cx_straight=[110,280,450]
cropping_cx_length_straight=170

cropping_cy_curved=[50,145,240,335]
cropping_cy_length_curved=95
cropping_cx_curved=[30,230,430]
cropping_cx_length_curved=200

#R.Pi pins and defaul states



def plcOutput(Pass):
    #Set R.P. pins as boolean outputs 
    #bit1=LED(14,False)
    #bit2=LED(15,False)
    #bit3=LED(18,False)
    #bit4=LED(23) 
    
    if Pass:
        #bit1.on()
        #bit2.off()
        #bit3.off()
        print("Good batch")
        
    else:
        #bit1.off()
        #bit2.off()
        #bit3.off()
        print("Batch rejected")
        
    return 0
    
    
"""   
def plcInout():
    #Set pins as boolean inputs
    #bit1=Button(5,False)
    #bit2=Button(6,False)
    #bit3=Button(13,False)
    
    if():
        return 
    
"""
def testCamera(img):
    for itemx in cropping_cx_straight:
        for itemy in cropping_cy_straight:
            img=cv.rectangle(img,(itemx,itemy),(itemx + cropping_cx_length_straight, itemy + cropping_cy_length_straight),(255,0,0),5)
    img=cv.rectangle(img,(20,180),(100,350),(255,0,0),5)
    plt.figure(figsize = (ds,ds))
    plt.imshow(img)
    plt.axis('off')
    plt.show()  
    
def testCameraCurved(img):
    for itemx in cropping_cx_curved:
        for itemy in cropping_cy_curved:
            img=cv.rectangle(img,(itemx,itemy),(itemx + cropping_cx_length_curved, itemy + cropping_cy_length_curved),(255,0,0),5)
    plt.figure(figsize = (ds,ds))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def cropStraightImage(img):
    position_x=0
    position_y=0
    Pass=True
    for itemx in cropping_cx_straight:
        position_x += 1
        for itemy in cropping_cy_straight:
            position_y %= 4
            position_y += 1
            img_cropped=img[itemy:itemy+cropping_cy_length_straight,itemx:itemx+cropping_cx_length_straight]
            avg_color_per_row = np.average(img_cropped, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            if avg_color[0]>80 and avg_color[1]>80 and avg_color[2]>80:
                tag, goodPart = Check(img_cropped,1,0,0)
                if goodPart == False:
                    Pass = False
            else:
                tag="Empty"
                goodPart = False
                Pass = False
            print("Y: " + str(position_y) + "; X: " + str(position_x) + "; Tag: " + str(tag) + "; Pass: " + str(goodPart))
    return Pass
    
def cropCurvedImage(img,isLeft):
    position_x=0
    position_y=0
    Pass=True
    for itemx in cropping_cx_curved:
        position_x += 1
        for itemy in cropping_cy_curved:
            position_y %= 4
            position_y += 1
            img_cropped=img[itemy:itemy+cropping_cy_length_curved,itemx:itemx+cropping_cx_length_curved]
            avg_color_per_row = np.average(img_cropped, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            if avg_color[0]>80 and avg_color[1]>80 and avg_color[2]>80:
                tag, goodPart = Check(img_cropped,0,isLeft, not isLeft)
                if goodPart == False:
                    Pass = False
            else:
                tag="Empty"
                goodPart = False
                Pass = False
            print("Y: " + str(position_y) + "; X: " + str(position_x) + "; Tag: " + str(tag) + "; Pass: " + str(goodPart))
    return Pass

def captureImage(img_count):
    
    cam = cv.VideoCapture(0)
    cv.namedWindow("test")
    img_counter = img_count


    while True: 
        ret, frame = cam.read()
        #frame = cv.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
        if not ret:
            print("failed to grab frame")
            break
        cv.imshow("test", frame)
        #out.write(frame)
                
        k = cv.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "./group5_test_images/opencv_frame_{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            
            cam.release()
            #out.release()
            cv.destroyAllWindows()
            return 0
            
    cam.release()
    #out.release()
    cv.destroyAllWindows()

    return 0   
        

        
def FeatureExtraction(img_rgb,contour_filter,contourList,tagList,isStraight,hole):
    img_feature=img_rgb.copy()
    
    #for i,contour in enumerate(contour_filter):
    for i in range(1):
        contour = contour_filter[0]
        tag=""
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
        
        if hole:
            tag="Hole inside"
        
        if tag=="":
            if isStraight:            
                if perimeter<400 and perimeter>200 and area<6000 and area>4000 and aspect_ratio<0.6 and aspect_ratio>0.35:
                    tag = "Train"
        
        if tag=="":
            if isStraight:
                if area < 0.70 * minStraightArea and perimeter < 0.70 * minStraightPerimeter:
                    tag="Cut in half"
            else:
                if area < 0.70 * minCurvedArea and perimeter < 0.90 * minCurvedPerimeter:
                    tag="Cut in half"
        
        if tag=="":
            for item in contourList:
                if isStraight:
                    match_score = cv.matchShapes(contour, item, cv.CONTOURS_MATCH_I3,0)
                else:
                    match_score = cv.matchShapes(contour, item, cv.CONTOURS_MATCH_I1,0)
                match_score_list.append(match_score)
            if area<500:
                tag= "Defect: Hole inside"
            else:
                tag=tagList[np.argmin(match_score_list)]
    
        img = cv.putText(img_feature,tag,(cx-50,cy+35),cv.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,255),1,cv.LINE_AA)
        
    
    # Display extracted features
    #plt.figure(figsize = (ds,ds))
    #plt.imshow(img_feature)
    #plt.axis('off')
    #plt.show()    
    if tag == "Good Part":
        goodPart = True
    else:
        goodPart = False
    return tag, goodPart

def FilterContours(img_rgb,contours,hierarchy):
    contour_filter=[]
    hole=False
    img_contour=img_rgb.copy()
    for i,contour in enumerate(contours):
        #Remove inside contours
        if hierarchy[0][i][3]==-1:
            #Remove small and large contours
            if cv.contourArea(contour) > 1500 and cv.contourArea(contour)<15000:
                cv.drawContours(img_contour, [contour], -1, (0,0,255), 2) 
                contour_filter.append(contour)
            #Find holes inside parts    
        if hierarchy[0][i][3]!=-1 and cv.contourArea(contour) > 100:
            hole=True
            cv.drawContours(img_contour, [contour], -1, (0,0,255), 2)
    return contour_filter, hole

def getShape(img):
    ksize=5
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    img_grey_blur = cv.GaussianBlur(img_grey,(ksize,ksize),0)
    ret,img_otsu = cv.threshold(img_grey_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) 
    contours,hierarchy = cv.findContours(img_otsu,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        if hierarchy[0][i][3]==-1:
            return contour

def readContours(isStraight):
    contourList = []
    if isStraight:
        imgStraight = cv.imread("./shapes/straight_shape.png")
        imgDefect1 = cv.imread("./shapes/straight_defect_1.png")
        imgDefect2 = cv.imread("./shapes/straight_defect_2.png")
        imgDefect3 = cv.imread("./shapes/straight_defect_3.png")
        straightContour = getShape(imgStraight)
        defect1Contour =  getShape(imgDefect1)
        defect2Contour = getShape(imgDefect2)
        defect3Contour = getShape(imgDefect3)
        contourList.append(straightContour)
        contourList.append(defect1Contour)
        contourList.append(defect2Contour)
        contourList.append(defect3Contour)
        tagList = ["Good Part", "Defect: Head Cut Off","Defect: Head Cut Off + Filled","Defect: Filled in"]
    else:
        imgCurved = cv.imread("./shapes/curved_shape.png")
        imgStraight = cv.imread("./shapes/straight_shape.png")
        imgDefect1 = cv.imread("./shapes/curved_defect_1.png")
        
        curvedContour = getShape(imgCurved)
        straightContour = getShape(imgStraight)
        defect1Contour =  getShape(imgDefect1)
        
        contourList.append(curvedContour)
        contourList.append(straightContour)
        contourList.append(defect1Contour)
        
        tagList=["Good Part", "Defect: Wrong Shape","Defect: Filled in"]
    return contourList, tagList

def getContour(img_rgb):
    img_grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    img_grey_blur = cv.GaussianBlur(img_grey,(ksize,ksize),0)
    ret,img_otsu = cv.threshold(img_grey_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    img_pro=cv.morphologyEx(img_otsu, cv.MORPH_OPEN, kernel)
    return cv.findContours(img_pro,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#Main Function
def Check(img, isStraight, isLeft, isRight):
    contourList,tagList = readContours(isStraight)
    if isRight:
        img=cv.flip(img, 0)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    contours, hierarchy = getContour(img_rgb)
    contour_filter,hole = FilterContours(img_rgb, contours, hierarchy)
    return FeatureExtraction(img_rgb,contour_filter,contourList,tagList,isStraight,hole)


 
for i in range (4):
   # Initialize pass value (default true) 
   Pass=True 
   captureImage(i+4)
   img=cv.imread("./group5_test_images/opencv_frame_"+str(i+4)+".png") 
   #img=cv.imread("./group5_defects_2/opencv_frame_"+str(i+8)+".png")
   img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   print("image read")
   plt.figure(figsize = (ds,ds))
   plt.imshow(img_rgb)
   plt.axis('off')
   plt.show()       
   print("------------------------------------")
   
   
   Pass=cropStraightImage(img_rgb)
   #Pass=cropCurvedImage(img_rgb, True)
   testCamera(img_rgb)
   #testCameraCurved(img_rgb)
   print(Pass)
   plcOutput(Pass)
   
""" 
   
   
   isStraight,isLeft,isRight = plsInput()

   #if(isStraight):
       #Pass=cropStraightImage(img_rgb)
    elif(isLeft):
        Pass=cropCurvedImage(img_rgb, True)
    elif(isRight):   
        Pass=cropCurvedImage(img_rgb, False)
        
"""    
      