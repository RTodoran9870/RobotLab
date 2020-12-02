# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:13:06 2020

@author: RTodo
"""
import numpy as np
import cv2 as cv

cropping_cy_straight=[35,130,225,320]
cropping_cy_length_straight=95
cropping_cx_straight=[110,280,450]
cropping_cx_length_straight=170

cropping_cy_curved=[50,145,240,335]
cropping_cy_length_curved=95
cropping_cx_curved=[30,230,430]
cropping_cx_length_curved=200

def testCamera(img):
    for itemx in cropping_cx_straight:
        for itemy in cropping_cy_straight:
            img=cv.rectangle(img,(itemx,itemy),(itemx + cropping_cx_length_straight, itemy + cropping_cy_length_straight),(255,0,0),5)
    img=cv.rectangle(img,(20,180),(100,350),(255,0,0),5)
    return img


cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    frame = testCamera(frame)

    # Display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()