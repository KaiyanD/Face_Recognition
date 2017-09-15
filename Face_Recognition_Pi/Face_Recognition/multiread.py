import cv2
import multiprocessing
import random
from time import sleep
import face_recognition
import os
import numpy as np
from multiprocessing import Manager, Pool

Kaiyan_image = face_recognition.load_image_file("Kaiyan.jpg")
Kaiyan_face_encoding = face_recognition.face_encodings(Kaiyan_image)[0]
Alex_image = face_recognition.load_image_file("Alex.jpg")
Alex_face_encoding = face_recognition.face_encodings(Alex_image)[0]
Julie_image = face_recognition.load_image_file("Julie.jpg")
Julie_face_encoding = face_recognition.face_encodings(Julie_image)[0]
Steve_image = face_recognition.load_image_file("Steve.jpg")
Steve_face_encoding = face_recognition.face_encodings(Steve_image)[0]
#Chandan_image = face_recognition.load_image_file("Chandan.jpg")
#Chandan_face_encoding = face_recognition.face_encodings(Chandan_image)[0]
#Op_image = face_recognition.load_image_file("Op.jpg")
#Op_face_encoding = face_recognition.face_encodings(Op_image)[0]
Morgane_image = face_recognition.load_image_file("Morgane.jpg")
Morgane_face_encoding = face_recognition.face_encodings(Morgane_image)[0]
Andrew_image = face_recognition.load_image_file("Andrew.jpg")
Andrew_face_encoding = face_recognition.face_encodings(Andrew_image)[0]
global Face_encoding_ls
Face_encoding_ls = [Kaiyan_face_encoding,Alex_face_encoding,Julie_face_encoding,Steve_face_encoding,Morgane_face_encoding,Andrew_face_encoding]

#faceCascade = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml")

def locateface(frame):
    face_locations = face_recognition.face_locations(frame)
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #face_locations = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors = 5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    #face_locations = np.array(face_locations).tolist()
    return(face_locations)
    print(face_locations)


def identiface(frame, locations, Face_encoding_ls):
    #face_locations = face_recognition.face_locations(frame)
    #face_encodings = face_recognition.face_encodings(frame, face_locations)
    names = []
    for location in locations:
        face_encoding = face_recognition.face_encodings(frame,[location])[0]
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([(Face_encoding_ls[0])], face_encoding)
        match1 = face_recognition.compare_faces([(Face_encoding_ls[1])], face_encoding)
        match2 = face_recognition.compare_faces([(Face_encoding_ls[2])], face_encoding)
        match3 = face_recognition.compare_faces([(Face_encoding_ls[3])], face_encoding)
        #match4 = face_recognition.compare_faces([(Face_encoding_ls[4])], face_encoding)
        #match5 = face_recognition.compare_faces([Op_face_encoding], face_encoding)
        match6 = face_recognition.compare_faces([(Face_encoding_ls[4])], face_encoding)
        match7 = face_recognition.compare_faces([(Face_encoding_ls[5])], face_encoding)
        name = "Unknown"
        if match[0]:
            name = "Ding"
        if match3[0]:
            name = "Steve"
        #if match4[0]:
            #name = "Chandan"
        if match1[0]:
            name = "Alex"
        if match2[0]:
            name = "Julie"
        #if match5[0]:
        #name = "Op"
        if match6[0]:
            name = "Morgane"
        if match7[0]:
            name = "Andrew"
        names.append(name)
    return(names)
    print(names)

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue, face_ls):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.face_ls = face_ls
    # Other initialization stuff
    def run(self):
        while True:
            frameNum,frameData = self.task_queue.get()
            # print(frameData)
            locations = face_recognition.face_locations(frameData)
            #encodings = face_recognition.face_encodings(frameData, locations)
            # print(encodings)
            #print(locations)
            #print(self.face_ls[0])
            # names = identiface(frameData,locations,self.face_ls)
            # names = identiface(frameData,locations,self.face_ls)
            #print(names)
            # print(frameNum)
            # locations = locateface(frameData)
            # names = identiface(frameData,face_locations)
            # locateface(frameData)
            # print(locateface(frameData))
            # locateface(frameData,self.result_queue)
            # Do computations on image
            #face_locations = faceCascade.detectMultiScale(
            #frameData,
            #scaleFactor=1.1,
            #minNeighbors = 5,
            #minSize=(80,80),
            #flags=cv2.CASCADE_SCALE_IMAGE
            #)
            #face_locations = np.array(face_locations).tolist()
            # Put result in queue
            #self.result_queue.put(str(face_locations))
            self.result_queue.put(locations)
        
        return



# No more than one pending task
tasks = multiprocessing.Queue(4)
results = multiprocessing.Queue()
# Init and start consumer
consumer1 = Consumer(tasks,results,Face_encoding_ls)
consumer2 = Consumer(tasks,results,Face_encoding_ls)
consumer3 = Consumer(tasks,results,Face_encoding_ls)
#consumer4 = Consumer(tasks,results,Face_encoding_ls)

consumer1.start()
consumer2.start()
consumer3.start()
#consumer4.start()
#consumers = [consumer1,consumer2,consumer3,consumer4]

#Creating window and starting video capturer from camera
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
#Try to get the first frame
if vc.isOpened():
    rval, frame = vc.read()
    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
else:
    rval = False

# Dummy int to represent frame number for display
frameNum = 0
# String for result
text = str(None)
locations = []
font = cv2.FONT_HERSHEY_SIMPLEX

# Process loop
while rval:
    # Grab image from stream
    frameNum += 1
    # Put image in task queue if empty
    try:
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        tasks.put_nowait((frameNum, frame))
    except:
        pass
    # Get result if ready
    try:
        # Use this if processing is fast enough
        # text = results.get(timeout=0.4)
        # Use this to prefer smooth display over frame/text shift
        locations = results.get_nowait()
    except:
        pass
    
    # Add last available text to last image and display
    # Showing the frame with all the applied modifications
    if len(locations) > 0:
        encodings = face_recognition.face_encodings(frame, locations)
        for (top, right, bottom, left), face_encoding in zip(locations, encodings):
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces([Kaiyan_face_encoding], face_encoding)
            match1 = face_recognition.compare_faces([Alex_face_encoding], face_encoding)
            match2 = face_recognition.compare_faces([Julie_face_encoding], face_encoding)
            match3 = face_recognition.compare_faces([Steve_face_encoding], face_encoding)
            #match4 = face_recognition.compare_faces([Chandan_face_encoding], face_encoding)
            #match5 = face_recognition.compare_faces([Op_face_encoding], face_encoding)
            match6 = face_recognition.compare_faces([Morgane_face_encoding], face_encoding)
            match7 = face_recognition.compare_faces([Andrew_face_encoding], face_encoding)
        
            name = "Unknown"
            if match[0]:
                name = "Ding"
            if match3[0]:
                name = "Steve"
            #if match4[0]:
                #name = "Chandan"
            if match1[0]:
                name = "Alex"
            if match2[0]:
                name = "Julie"
            #if match5[0]:
                #name = "Op"
            if match6[0]:
                name = "Morgane"
            if match7[0]:
                name = "Andrew"
    
    
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("preview", frame)
    # Getting next frame from camera
    rval, frame = vc.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Optional image resize
# frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
cv2.destroyAllWindows()
vc.release()
consumer1.terminate()
consumer2.terminate()
consumer3.terminate()
#consumer4.terminate()