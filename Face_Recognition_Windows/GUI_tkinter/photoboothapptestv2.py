# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import Tkinter as tki
import threading
import datetime
import imutils
import cv2
import os

whichfilter = 0
# xml files describing our haar cascade classifiers
faceCascadeFilePath = "haarcascade_frontalface_default.xml"
noseCascadeFilePath = "haarcascade_mcs_nose.xml"
eyeCascadeFilePath = "haarcascade_eye_tree_eyeglasses.xml"

# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
eyeCascade = cv2.CascadeClassifier(eyeCascadeFilePath)

#-----------------------------------------------------------------------------
#       Load and configure mustache (.png with alpha transparency)
#-----------------------------------------------------------------------------
 
# Load our overlay image: mustache.png
imgMustache = cv2.imread('mustache.png',-1)
 
# Create the mask for the mustache
orig_Mustachemask = imgMustache[:,:,3]
 
# Create the inverted mask for the mustache
orig_Mustachemask_inv = cv2.bitwise_not(orig_Mustachemask)
 
# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

imgSunglass = cv2.imread('sunglass1.png',-1)
 
# Create the mask for the mustache
orig_Sunglassmask = imgSunglass[:,:,3]
 
# Create the inverted mask for the mustache
orig_Sunglassmask_inv = cv2.bitwise_not(orig_Sunglassmask)
 
# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgSunglass = imgSunglass[:,:,0:3]
origSunglassHeight, origSunglassWidth = imgSunglass.shape[:2]

class PhotoBoothApp:
	def __init__(self, vs, outputPath):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.outputPath = outputPath
		self.frame = None
		self.thread = None
		self.stopEvent = None
                self.captured = None
		# initialize the root window and image panel
		self.root = tki.Tk()
		self.panel = None

		# create a button, that when pressed, will take the current
		# frame and save it to file
		btn1 = tki.Button(self.root, text="Snapshot!",
			command=self.snapshot)
		btn1.pack(side="bottom", padx=10,
			pady=10)
		frames = tki.Frame(self.root)
		frames.pack(side = "bottom",fill = "both", expand="yes")
		btn2 = tki.Button(frames, text="No filters",
			command=self.add_Nothing)
		btn2.pack(side="left",fill="both",expand="yes", padx=10,
			pady=10)
		btn3 = tki.Button(frames, text="Mustache",
			command=self.add_Mustache)
		btn3.pack(side="left",fill="both",expand="yes", padx=10,
			pady=10)
		btn4 = tki.Button(frames, text="Sunglass",
			command=self.add_Sunglass)
		btn4.pack(side="left",fill="both",expand="yes", padx=10,
			pady=10)

		# start a thread that constantly pools the video sensor for
		# the most recently read frame
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("PhotoBooth")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

	def videoLoop(self):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
		try:   
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				# grab the frame from the video stream and resize it to
				# have a maximum width of 300 pixels
				self.frame = self.vs.read()
				if whichfilter == 1:
				    frame = self.frame
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
     
                                            # Detect a nose within the region bounded by each face (the ROI)
                                            nose = noseCascade.detectMultiScale(roi_gray)
     
                                            for (nx,ny,nw,nh) in nose:
                                                    # Un-comment the next line for debug (draw box around the nose)
                                                    # cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
     
                                                    # The mustache should be three times the width of the nose
                                                    mustacheWidth =  3 * nw
                                                    mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth
  
                                                    # Center the mustache on the bottom of the nose
                                                    x1 = nx - (mustacheWidth/4)
                                                    x2 = nx + nw + (mustacheWidth/4)
                                                    y1 = ny + nh - (mustacheHeight/2)
                                                    y2 = ny + nh + (mustacheHeight/2)
    
                                                    # Check for clipping
                                                    if x1 < 0:
                                                            x1 = 0
                                                    if y1 < 0:
                                                            y1 = 0
                                                    if x2 > w:
                                                            x2 = w
                                                    if y2 > h:
                                                            y2 = h
 
                                                    # Re-calculate the width and height of the mustache image
                                                    mustacheWidth = x2 - x1
                                                    mustacheHeight = y2 - y1
 
                                                    # Re-size the original image and the masks to the mustache sizes
                                                    # calcualted above
                                                    mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
                                                    mask = cv2.resize(orig_Mustachemask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
                                                    mask_inv = cv2.resize(orig_Mustachemask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
     
                                                    # take ROI for mustache from background equal to size of mustache image
                                                    roi = roi_color[y1:y2, x1:x2]
       
                                                    # roi_bg contains the original image only where the mustache is not
                                                    # in the region that is the size of the mustache.
                                                    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    
                                                    # roi_fg contains the image of the mustache only where the mustache is
                                                    roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
    
                                                    # join the roi_bg and roi_fg
                                                    dst = cv2.add(roi_bg,roi_fg)
     
                                                    # place the joined image, saved to dst back over the original image
                                                    roi_color[y1:y2, x1:x2] = dst

                                                    self.frame = frame
                                                    break
                                if whichfilter == 2:
				    frame = self.frame
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
     
                                            # Detect a nose within the region bounded by each face (the ROI)
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
                                                    mask = cv2.resize(orig_Sunglassmask, (SunglassWidth,SunglassHeight), interpolation = cv2.INTER_AREA)
                                                    mask_inv = cv2.resize(orig_Sunglassmask_inv, (SunglassWidth,SunglassHeight), interpolation = cv2.INTER_AREA)
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
                                                    self.frame = frame
                                                    
                                if whichfilter == 3:
                                    self.frame = cv2.imread("captured.jpg")

	         		#self.frame = imutils.resize(self.frame, width=300)
		                self.captured = self.frame
		        	# OpenCV represents images in BGR order; however PIL
			        # represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				image = ImageTk.PhotoImage(image)
				
				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tki.Label(image=image)
					self.panel.image = image
					self.panel.pack(side="left", padx=10, pady=10)
		
				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image
                    
		except RuntimeError, e:
			print("[INFO] caught a RuntimeError")
        
	def takeSnapshot(self):
		# grab the current timestamp and use it to construct the
		# output path
		ts = datetime.datetime.now()
		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		p = os.path.sep.join((self.outputPath, filename))
		# save the file
		cv2.imwrite(p, self.frame.copy())
		print("[INFO] saved {}".format(filename))
        def snapshot(self):
                cv2.imwrite("captured.jpg",self.captured.copy())
                global whichfilter
                whichfilter = 3
        def add_Mustache(self):
                global whichfilter
                whichfilter = 1
        def add_Sunglass(self):
                global whichfilter
                whichfilter = 2
        def add_Nothing(self):
                global whichfilter
                whichfilter = 0
	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()
