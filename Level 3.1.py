# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:57:58 2020

@author: RTodo
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from save_results import ResultsSave
from time import sleep

#Configure Raspberry Pi GPIO
import RPi.GPIO as GPIO

#Set GPIO port numbering
GPIO.setmode(GPIO.BCM)               # BCM for GPIO numbering  

#Set input pins
GPIO.setup(26, GPIO.IN,  pull_up_down=GPIO.PUD_DOWN) # input1
GPIO.setup(20, GPIO.IN,  pull_up_down=GPIO.PUD_DOWN) # input2
GPIO.setup(21, GPIO.IN,  pull_up_down=GPIO.PUD_DOWN) # input3
 
#Set output pins
GPIO.setup(5, GPIO.OUT, initial=1)    # Output 1
GPIO.setup(12, GPIO.OUT, initial=1)    # Output 2
GPIO.setup(6, GPIO.OUT, initial=1)    # Output 3
GPIO.setup(13, GPIO.OUT, initial=1)    # Output 4 (Relay 1)
GPIO.setup(19, GPIO.OUT, initial=1)    # Output 5 (Relay 2)
GPIO.setup(16, GPIO.OUT, initial=1)    # Output 6 (Relay 3)

tray_counter = 0
ds=12
#Precalibrating ethalons
minStraightArea = 5000
minStraightPerimeter = 350
minCurvedArea = 8879.5
minCurvedPerimeter = 541.65
ksize=3
itern=2
kernel = np.ones((ksize,ksize),np.uint8)

#Cropping settings
cropping_cy_straight=[35,130,225,320]
cropping_cy_length_straight=95
cropping_cx_straight=[100,270,440]
cropping_cx_length_straight=190

cropping_cy_curved=[50,145,240,335]
cropping_cy_length_curved=95
cropping_cx_curved=[20,220,420]
cropping_cx_length_curved=220


class Part:
    def __init__(self, position_x,position_y,isCorrect,description):
        self.position_x = position_x
        self.position_y = position_y
        self.isCorrect = isCorrect
        self.description = description
        
def displayGPIO():
    # Get input pin values and display them 
    pins_input_List=[]
    pins_input_List.append(GPIO.input(26))
    pins_input_List.append(GPIO.input(20))
    pins_input_List.append(GPIO.input(21))
    print("Inputs: "+str(pins_input_List))
    
    # Get output pin values and display them
    pins_output_List=[]
    
    pins_output_List.append(GPIO.input(5))
    pins_output_List.append(GPIO.input(12))
    pins_output_List.append(GPIO.input(6))
    pins_output_List.append(GPIO.input(13))
    pins_output_List.append(GPIO.input(19))
    pins_output_List.append(GPIO.input(16))
    print("Outputs: "+str(pins_output_List))
    
    return pins_input_List,pins_output_List        

def plcOutput(partList,batch):
    sleep(2.5)   
    my_results_plc=ResultsSave('group5_vision_result_dummy'+str(batch)+'.csv','group5_plc_result_lvl3'+str(batch)+'.csv')
    
    for row in range(4):
        #Set bits corresponding to correct rows
        if row==0:
            #specify row 1
            GPIO.output(19,1)
            GPIO.output(16,1)
            
            #send quality bits
            for part in partList:
                if part.position_y==1 and part.position_x==1 and part.isCorrect==True:
                    GPIO.output(5,0)
                elif part.position_y==1 and part.position_x==2 and part.isCorrect==True:
                    GPIO.output(12,0)
                elif part.position_y==1 and part.position_x==3 and part.isCorrect==True:
                    GPIO.output(5,0) 
            
            #Send pulse relay 1 that row is done
            sleep(0.5)
            GPIO.output(13, 0)     # Finished inspection on batch row - Relay 1 - up 
            sleep(0.5)
            
            #Display and save results for row
            pins_input_List,pins_output_List = displayGPIO()
            response = str(pins_input_List)+str(pins_output_List)
            my_results_plc.insert_plc(batch,response)
            
            # Reset relay 1 after 0.5s after each row is inspected
            GPIO.output(13,1)
            
        elif row==1: 
            #specify row 2
            GPIO.output(19,1)
            GPIO.output(16,0)
            
            #send quality bits
            for part in partList:
                if part.position_y==2 and part.position_x==1 and part.isCorrect==True:
                    GPIO.output(5,0)
                elif part.position_y==2 and part.position_x==2 and part.isCorrect==True:
                    GPIO.output(12,0)
                elif part.position_y==2 and part.position_x==3 and part.isCorrect==True:
                    GPIO.output(5,0)
            
            #Send pulse relay 1 that row is done
            sleep(0.5)
            GPIO.output(13, 0)     # Finished inspection on batch row - Relay 1 - up 
            sleep(0.5)
            
            #Display and save results for row
            pins_input_List,pins_output_List = displayGPIO()
            response = str(pins_input_List)+str(pins_output_List)
            my_results_plc.insert_plc(batch,response)
            
            # Reset relay 1 after 0.5s after each row is inspected
            GPIO.output(13,1)
            
        elif row==2: 
            #specify row 3
            GPIO.output(19,0)
            GPIO.output(16,1)
            #send quality bits
            for part in partList:
                if part.position_y==3 and part.position_x==1 and part.isCorrect==True:
                    GPIO.output(5,0)
                elif part.position_y==3 and part.position_x==2 and part.isCorrect==True:
                    GPIO.output(12,0)
                elif part.position_y==3 and part.position_x==3 and part.isCorrect==True:
                    GPIO.output(5,0) 
                    
            #Send pulse relay 1 that row is done
            sleep(0.5)
            GPIO.output(13, 0)     # Finished inspection on batch row - Relay 1 - up 
            sleep(0.5)
            
            #Display and save results for row
            pins_input_List,pins_output_List = displayGPIO()
            response = str(pins_input_List)+str(pins_output_List)
            my_results_plc.insert_plc(batch,response)
            
            # Reset relay 1 after 0.5s after each row is inspected
            GPIO.output(13,1)
            
        elif row==3:
            #specify row 4
            GPIO.output(19,0)
            GPIO.output(16,0)
            #send quality bits
            for part in partList:
                if part.position_y==4 and part.position_x==1 and part.isCorrect==True:
                    GPIO.output(5,0)
                elif part.position_y==4 and part.position_x==2 and part.isCorrect==True:
                    GPIO.output(12,0)
                elif part.position_y==4 and part.position_x==3 and part.isCorrect==True:
                    GPIO.output(5,0) 
           
            #Send pulse relay 1 that row is done
            sleep(0.5)
            GPIO.output(13, 0)     # Finished inspection on batch row - Relay 1 - up 
            sleep(0.5)
            
            #Display and save results for row
            pins_input_List,pins_output_List = displayGPIO()
            response = str(pins_input_List)+str(pins_output_List)
            my_results_plc.insert_plc(batch,response)
            
            # Reset relay 1 after 0.5s after each row is inspected
            GPIO.output(13,1)
                       

    return 0
    
    
def plcInput(img_num):
    
    # If signal from plc recieved, begin inspection
    if GPIO.input(26):
        # Begin inspection on batct
        partList = videoCapture()
        
        print(partList)
        plcOutput(partList,img_num)   #Call plcOutput fn
        img_num+=1
        
    else:
        print("Waiting for signal from PLC.")
        sleep(0.25)
        print("Waiting for signal from PLC..")
        sleep(0.25)
        print("Waiting for signal from PLC...")
    
    #sleep(0.5) #time lag for cheching the message from PLC
    return img_num
        
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
    global  tray_counter
    tray_counter += 1
    partList = []
    position_x=0
    position_y=0
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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
                  
            else:
                tag="Empty"
                goodPart = False

            print("Y: " + str(position_y) + "; X: " + str(position_x) + "; Tag: " + str(tag) + "; Pass: " + str(goodPart))
            part = Part(position_x, position_y, isCorrect=goodPart, description=tag)
            partList.append(part)
    #Save results
    my_results_vision=ResultsSave('group5_vision_result_lvl3'+str(tray_counter)+'.csv','group5_plc_result_dummy'+str(tray_counter)+'.csv')
    for part in partList:
        my_results_vision.insert_vision(tray_counter,part.position_x+3*(part.position_y-1),'straight',part.isCorrect,part.description)
        
    return partList
    


def cropCurvedImage(img,isLeft):
    global  tray_counter
    tray_counter += 1
    partList = []
    position_x=0
    position_y=0
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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
                
            else:
                tag="Empty"
                goodPart = False
            print("Y: " + str(position_y) + "; X: " + str(position_x) + "; Tag: " + str(tag) + "; Pass: " + str(goodPart))
            part = Part(position_x, position_y, isCorrect=goodPart, description=tag)
            partList.append(part)
    #Save results
    my_results=ResultsSave('group5_vision_result_lvl2'+str(tray_counter)+'.csv','group5_plc_result_dummy.csv')
    for part in partList:
        my_results.insert_vision(tray_counter,part.position_x+3*(part.position_y-1),'curved',part.isCorrect,part.description)
    
    return partList

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
        imgDefect2 = cv.imread("./shapes/curved_defect_2.png")
        imgDefect3 = cv.imread("./shapes/curved_defect_3.png")
        imgDefect3 = cv.flip(imgDefect3,0)
        
        
        curvedContour = getShape(imgCurved)
        straightContour = getShape(imgStraight)
        defect1Contour =  getShape(imgDefect1)
        defect2Contour = getShape(imgDefect2)
        defect3Contour = getShape(imgDefect3)
        
        contourList.append(curvedContour)
        contourList.append(straightContour)
        contourList.append(defect1Contour)
        contourList.append(defect2Contour)
        contourList.append(defect3Contour)
        
        tagList=["Good Part", "Defect: Wrong Shape","Defect: Filled in","Defect: Filled in + Head Cut off", "Defect: Head Cut off"]
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

def readContoursType():
    contourList = []
    imgCurved = cv.imread("./shapes/curved_shape.png")
    imgStraight = cv.imread("./shapes/Shapes_Moving/straight_shape.jpeg")
        
        
    curvedContour = getShape(imgCurved)
    straightContour = getShape(imgStraight)
        
    contourList.append(curvedContour)
    contourList.append(straightContour)

        
    tagList=["Straight","Shape"]
    return contourList, tagList

def FilterContoursType(img_rgb,contours,hierarchy):
    contour_filter=[]
    img_contour=img_rgb.copy()
    for i,contour in enumerate(contours):
        #Remove inside contours
        if hierarchy[0][i][3]==-1:
            #Remove small and large contours
            if cv.contourArea(contour) > 1500 and cv.contourArea(contour)<15000:
                cv.drawContours(img_contour, [contour], -1, (0,0,255), 2)  
                contour_filter.append(contour)
    plt.figure(figsize = (ds,ds))
    plt.imshow(img_contour)
    plt.axis('off')
    plt.show()
    return contour_filter



def CheckType(img):
    straight_score = 0
    curved_score = 0
    contourList,tagList = readContoursType()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    contours, hierarchy = getContour(img_rgb)
    contours_filtered = FilterContoursType(img_rgb, contours, hierarchy)
    for contour in contours_filtered:
        straight_score += cv.matchShapes(contour, contourList[1], cv.CONTOURS_MATCH_I3,0)
        curved_score += cv.matchShapes(contour, contourList[0], cv.CONTOURS_MATCH_I3,0)
    if straight_score < curved_score:
        tag = "Straight"
    else: 
        tag = "Curved"
    return(tag)


def videoCapture():
    cap = cv.VideoCapture(0)
    flag = 0

    while(flag==0):
    # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #frame = cv.rectangle(frame,(5,160),(20,180),(255,0,0),2)
        frame_cropped = frame[160:180,5:20]


        avg_color_per_row = np.average(frame_cropped, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        #print(avg_color)
        if avg_color[0]<110:
            flag=1
            #print(avg_color)
    # Display the resulting frame
    """
    TODO in lab: Check how long we should sleep
    """
    sleep(0.5)
    
    """
    #Testing version (with video)
    for i in range(25):
        ret, frame = cap.read()
    """
    
    """
    TODO: save the image rather than send it to crop function (similar to level 1 and 2)
    """
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if CheckType(frame) == "Straight":
        partList, Pass=cropStraightImage(frame)
    else:
        partList, Pass=cropCurvedImage(frame)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    
    return partList
    
# Call functions
batch=1 
while True and batch<=20:
    k = cv.waitKey(5)
    
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
    else:
        batch = plcInput(batch)

print("Inspection of all batches complete")

# Clean up pins 
GPIO.cleanup()     