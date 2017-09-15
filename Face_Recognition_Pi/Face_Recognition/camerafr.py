"""
Created on Thu Mar 23 09:20:34 2017

@author: keynes
"""

import face_recognition
import cv2


# This is a super simple demo of running face recognition on live video from your webcam.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
Kaiyan_image = face_recognition.load_image_file("Kaiyan.jpg")
Kaiyan_face_encoding = face_recognition.face_encodings(Kaiyan_image)[0]
Alex_image = face_recognition.load_image_file("Alex.jpg")
Alex_face_encoding = face_recognition.face_encodings(Alex_image)[0]
Julie_image = face_recognition.load_image_file("Julie.jpg")
Julie_face_encoding = face_recognition.face_encodings(Julie_image)[0]
Steve_image = face_recognition.load_image_file("Steve.jpg")
Steve_face_encoding = face_recognition.face_encodings(Steve_image)[0]
#Jie_image = face_recognition.load_image_file("Jie.jpg")
#Jie_face_encoding = face_recognition.face_encodings(Jie_image)[0]
#Op_image = face_recognition.load_image_file("Op.jpg")
#Op_face_encoding = face_recognition.face_encodings(Op_image)[0]
Morgane_image = face_recognition.load_image_file("Morgane.jpg")
Morgane_face_encoding = face_recognition.face_encodings(Morgane_image)[0]
Andrew_image = face_recognition.load_image_file("Andrew.jpg")
Andrew_face_encoding = face_recognition.face_encodings(Andrew_image)[0]

faceCascadeFilePath = "lbpcascade_frontalface_improved.xml"
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_locations = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Find all the faces and face enqcodings in the frame of video
    # face_locations = face_recognition.face_locations(frame)
    # face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in this frame of video
    # for (x, y, w, h), face_encoding in zip(face_locations, face_encodings):
    for (x,y,w,h) in face_locations:
        # See if the face is a match for the known face(s)
        face_encoding = face_recognition.face_encodings(frame, [(y,x+w,y+h,x)])
        match = face_recognition.compare_faces([Kaiyan_face_encoding], face_encoding)
        match1 = face_recognition.compare_faces([Alex_face_encoding], face_encoding)
        match2 = face_recognition.compare_faces([Julie_face_encoding], face_encoding)
        match3 = face_recognition.compare_faces([Steve_face_encoding], face_encoding)
        #match4 = face_recognition.compare_faces([Jie_face_encoding], face_encoding)
        #match5 = face_recognition.compare_faces([Op_face_encoding], face_encoding)
        match6 = face_recognition.compare_faces([Morgane_face_encoding], face_encoding)
        match7 = face_recognition.compare_faces([Andrew_face_encoding], face_encoding)

        name = "Unknown"
        if match[0]:
            name = "Ding"
        if match3[0]:
            name = "Steve"
        #if match4[0]:
            #name = "Jie"
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
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), cv2.FILLED)
        #cv2.rectangle(frame,(h,w-35),(y,w),(0,0,255))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)
        print(name)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()