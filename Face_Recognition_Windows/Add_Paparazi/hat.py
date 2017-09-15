# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:14:13 2017

@author: kading
"""


import cv2  # OpenCV Library
 
#-----------------------------------------------------------------------------
#       Load and configure Haar Cascade Classifiers
#-----------------------------------------------------------------------------
 
# location of OpenCV Haar Cascade Classifiers:
#baseCascadePath = "C:\Users\kading\Downloads\data\haarcascades"

# xml files describing our haar cascade classifiers
faceCascadeFilePath = "haarcascade_frontalface_default.xml"
eyeCascadeFilePath = "haarcascade_eye_tree_eyeglasses.xml"

# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
eyeCascade = cv2.CascadeClassifier(eyeCascadeFilePath)
 
#-----------------------------------------------------------------------------
#       Load and configure mustache (.png with alpha transparency)
#-----------------------------------------------------------------------------
 
# Load our overlay image: hat.png
imgHat = cv2.imread('hat.png',-1)
 
# Create the mask for the hat
orig_mask = imgHat[:,:,3]
 
# Create the inverted mask for the hat
orig_mask_inv = cv2.bitwise_not(orig_mask)
 
# Convert hat image to BGR
# and save the original image size (used later when re-sizing the image)
imgHat = imgHat[:,:,0:3]
origHatHeight, origHatWidth = imgHat.shape[:2]

video_capture = cv2.VideoCapture(0)
while True:
    # Capture video feed
    ret, frame = video_capture.read()
    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in input video stream
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Iterate over each face found
    for (x, y, w, h) in faces:
        x1 = x-w/3
        x2 = x+w*4/3
        y1 = y-h*2/3
        y2 = y
        if y1 < 0:
            break
        HatWidth = x2 - x1
        HatHeight = y2 - y1
        Hat = cv2.resize(imgHat, (HatWidth,HatHeight), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(orig_mask, (HatWidth,HatHeight), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (HatWidth,HatHeight), interpolation = cv2.INTER_AREA)
        roi = frame[y1:y2, x1:x2]
        # roi_bg contains the original image only where the mustache is not
        # in the region that is the size of the mustache.
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        # roi_fg contains the image of the mustache only where the mustache is
        roi_fg = cv2.bitwise_and(Hat,Hat,mask = mask)
        # join the roi_bg and roi_fg
        dst = cv2.add(roi_bg,roi_fg)
        # place the joined image, saved to dst back over the original image
        frame[y1:y2, x1:x2] = dst
    # Display the resulting frame
    cv2.imshow('Video', frame)
    # press any key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
    
