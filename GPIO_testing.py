# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:23:17 2020

@author: Alexander
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from time import sleep


#Configure Raspberry Pi GPIO
import RPi.GPIO as GPIO


sleep(2)
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

"""    
def plcOutput(Pass):
      
    if Pass:
        GPIO.output(5, 0)     # Passed batch - Op1 - high
        GPIO.output(13, 0)     # Finished inspection on batch - Relay 1 - up
        print("Good batch")
        
        sleep(0.5)
        GPIO.output(5, 1)     # Reset op1 to defaul low value
        GPIO.output(13, 1)     # Reset relay to default low value
        
    else:
        GPIO.output(5, 1)     # Failed batch - Op1 - low
        GPIO.output(13, 0)     # Finished inspection on batch - Relay 1 - up
        print("Batch rejected")
    
        sleep(0.5)
        GPIO.output(13, 1)     # Reset relay to default low value
        
    return 0
    
    
  
def plcInput(img_num):
    
    if GPIO.input(26):
        # Begin inspection on batch
        Pass=True   # Initialize pass value (default true) 
        captureImage(img_num)    #Call capture image fn
        img=cv.imread("./group5_test_images/opencv_frame_"+str(img_num)+".png")     #Read captured image
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        
        print(Pass)
        plcOutput(Pass)
        img_num+=1
    
    sleep(0.5) #time lag for cheching the message from PLC
    return img_num
"""

displayGPIO()

sleep(2)
GPIO.output(5, 0)     # Reset op1 to defaul low value
GPIO.output(13, 0)

displayGPIO()


# Clean up pins 
GPIO.cleanup() 