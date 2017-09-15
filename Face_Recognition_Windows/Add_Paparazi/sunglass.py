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
 
# Load our overlay image: mustache.png
imgSunglass = cv2.imread('sunglass1.png',-1)
 
# Create the mask for the mustache
orig_mask = imgSunglass[:,:,3]
 
# Create the inverted mask for the mustache
orig_mask_inv = cv2.bitwise_not(orig_mask)
 
# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgSunglass = imgSunglass[:,:,0:3]
origSunglassHeight, origSunglassWidth = imgSunglass.shape[:2]

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
        # Un-comment the next line for debug (draw box around all faces)
        # face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
 
        # Detect eyes within the region bounded by each face (the ROI)
        eye = eyeCascade.detectMultiScale(roi_gray)
        if len(eye) == 2:
            [x1,y1,w1,h1] = eye[0]
            [x2,y2,w2,h2] = eye[1]
            nx = (x1+x2)/2
            ny = (y1+y2)/2
            nw = (w1+w2)/2
            nh = (h1+h2)/2
            # cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
            SunglassWidth =  3 * nw
            SunglassHeight = SunglassWidth * origSunglassHeight / origSunglassWidth
            # Center the mustache on the center bottom of eyes
            sgx1 = nx - (SunglassWidth)
            sgx2 = nx + nw + (SunglassWidth)
            sgy1 = ny - (SunglassHeight/4)
            sgy2 = ny + nh + (SunglassHeight/4)
            # Check for clipping
            if sgx1 < 0:
                sgx1 = 0
            if sgy1 < 0:
                sgy1 = 0
            if sgx2 > w:
                sgx2 = w
            if sgy2 > h:
                sgy2 = h
            # Re-calculate the width and height of the sunglass image
            SunglassWidth = sgx2 - sgx1
            SunglassHeight = sgy2 - sgy1
            # Re-size the original image and the masks to the mustache sizes
            # calcualted above
            Sunglass = cv2.resize(imgSunglass, (SunglassWidth,SunglassHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (SunglassWidth,SunglassHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (SunglassWidth,SunglassHeight), interpolation = cv2.INTER_AREA)
            # take ROI for mustache from background equal to size of mustache image
            roi = roi_color[sgy1:sgy2, sgx1:sgx2]
            # roi_bg contains the original image only where the mustache is not
            # in the region that is the size of the mustache.
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            # roi_fg contains the image of the mustache only where the mustache is
            roi_fg = cv2.bitwise_and(Sunglass,Sunglass,mask = mask)
            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg,roi_fg)
            # place the joined image, saved to dst back over the original image
            roi_color[sgy1:sgy2, sgx1:sgx2] = dst
           
    # Display the resulting frame
    cv2.imshow('Video', frame)
    # press any key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    cv2.imshow('Video', frame)
    # press any key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
