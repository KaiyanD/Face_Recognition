from multiprocessing import Pool, Manager
import cv2
import face_recognition
import scipy.misc
import dlib
import numpy as np
from time import time


t1 = time()

# Load a sample picture and learn how to recognize it.
Kaiyan_image = face_recognition.load_image_file("Kaiyan.jpg")
Kaiyan_face_encoding = face_recognition.face_encodings(Kaiyan_image)[0]
Alex_image = face_recognition.load_image_file("Alex.jpg")
Alex_face_encoding = face_recognition.face_encodings(Alex_image)[0]
Julie_image = face_recognition.load_image_file("Julie.jpg")
Julie_face_encoding = face_recognition.face_encodings(Julie_image)[0]
Steve_image = face_recognition.load_image_file("Steve.jpg")
Steve_face_encoding = face_recognition.face_encodings(Steve_image)[0]
Morgane_image = face_recognition.load_image_file("Morgane.jpg")
Morgane_face_encoding = face_recognition.face_encodings(Morgane_image)[0]
Andrew_image = face_recognition.load_image_file("Andrew.jpg")
Andrew_face_encoding = face_recognition.face_encodings(Andrew_image)[0]

manager = Manager()
jobs = manager.dict()
jobs["Kaiyan"] = cv2.imread("notcaptured.jpg")
jobs["Morgane"] = cv2.imread("notcaptured.jpg")
jobs["Alex"] = cv2.imread("notcaptured.jpg")
jobs["Steve"] = cv2.imread("notcaptured.jpg")
jobs["Andrew"] = cv2.imread("notcaptured.jpg")
jobs["Julie"] = cv2.imread("notcaptured.jpg")

jobs["Kaiyan_bm"] = 0.6
jobs["Morgane_bm"] = 0.6
jobs["Alex_bm"] = 0.6
jobs["Steve_bm"] = 0.6
jobs["Andrew_bm"] = 0.6
jobs["Julie_bm"] = 0.6
jobs["number"] = 1

def pick_frame(frls):
    print(jobs["number"])
    jobs["number"] += 1
    frame = frls[0]
    frame1 = frls[1]
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    # Loop through each face in this frame of video
    for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
        Kaiyan_dis = np.linalg.norm([Kaiyan_face_encoding]-face_encoding,axis=1)
        Morgane_dis = np.linalg.norm([Morgane_face_encoding]-face_encoding,axis=1)
        Alex_dis = np.linalg.norm([Alex_face_encoding]-face_encoding,axis=1)
        Steve_dis = np.linalg.norm([Steve_face_encoding]-face_encoding,axis=1)
        Andrew_dis = np.linalg.norm([Andrew_face_encoding]-face_encoding,axis=1)
        Julie_dis = np.linalg.norm([Julie_face_encoding]-face_encoding,axis=1)
        global Morgane,Alex,Steve,Andrew,Julie,Morgane_bm,Alex_bm,Steve_bm,Andrew_bm,Julie_bm
        if Kaiyan_dis < jobs["Kaiyan_bm"]:
            jobs["Kaiyan_bm"] = Kaiyan_dis
            jobs["Kaiyan"] = frame1
            #Kaiyan_in = t
        if Morgane_dis < jobs["Morgane_bm"]:
            jobs["Morgane_bm"] = Morgane_dis
            jobs["Morgane"] = frame1
	    #Morgane_in = t
        if Alex_dis < jobs["Alex_bm"]:
            jobs["Alex_bm"] = Alex_dis
            jobs["Alex"] = frame1
            #Alex_in = t
        if Steve_dis < jobs["Steve_bm"]:
            jobs["Steve_bm"] = Steve_dis
            jobs["Steve"] = frame1
            #Steve_in = t
        if Andrew_dis < jobs["Andrew_bm"]:
            jobs["Andrew_bm"] = Anrew_dis
            jobs["Andrew"] = frame1
            #Andrew_in = t
        if Julie_dis < jobs["Julie_bm"]:
            jobs["Julie_bm"] = Julie_dis
            jobs["Julie"] = frame1
            #Julie_in = t

frlss = []
cap = cv2.VideoCapture("my_video3.h264")
cap1 = cv2.VideoCapture("my_video.h264")
while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if ret == True:
	frlss.append([frame,frame1])
    else:
	break

if __name__ == '__main__':
    p = Pool(6)
    p.map(pick_frame,frlss)


cv2.imwrite("results/Kaiyan.jpg",jobs["Kaiyan"])
cv2.imwrite("results/Morgane.jpg",jobs["Morgane"])
cv2.imwrite("results/Alex.jpg",jobs["Alex"])
cv2.imwrite("results/Steve.jpg",jobs["Steve"])
cv2.imwrite("results/Andrew.jpg",jobs["Andrew"])
cv2.imwrite("results/Julie.jpg",jobs["Julie"])

print(jobs["Kaiyan_bm"])
print(jobs["Morgane_bm"])
print(jobs["Alex_bm"])
print(jobs["Steve_bm"])
print(jobs["Andrew_bm"])
print(jobs["Julie_bm"])
t2 = time()
t3 = t2 - t1
print(t3)
