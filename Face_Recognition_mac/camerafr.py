"""
Created on Thu Mar 23 09:20:34 2017

@author: keynes
"""

import face_recognition
import cv2
import os

# This is a super simple demo of running face recognition on live video from your webcam.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
Kaiyan_image = face_recognition.load_image_file("/Users/keynes/Documents/Kaiyan.jpg")
Kaiyan_face_encoding = face_recognition.face_encodings(Kaiyan_image)[0]
#Rama_image = face_recognition.load_image_file("/Users/keynes/Documents/Rama.jpg")
#Rama_face_encoding = face_recognition.face_encodings(Rama_image)[0]
#Ram_image = face_recognition.load_image_file("/Users/keynes/Documents/ram.jpg")
#Ram_face_encoding = face_recognition.face_encodings(Ram_image)[0]
#Alex_image = face_recognition.load_image_file("/Users/keynes/Documents/Alex.jpg")
#Alex_face_encoding = face_recognition.face_encodings(Alex_image)[0]
#Julie_image = face_recognition.load_image_file("/Users/keynes/Documents/Julie.jpg")
#Julie_face_encoding = face_recognition.face_encodings(Julie_image)[0]
#Steve_image = face_recognition.load_image_file("/Users/keynes/Documents/Steve.jpg")
#Steve_face_encoding = face_recognition.face_encodings(Steve_image)[0]
#Chandan_image = face_recognition.load_image_file("/Users/keynes/Documents/Chandan.jpg")
#Chandan_face_encoding = face_recognition.face_encodings(Chandan_image)[0]
#Op_image = face_recognition.load_image_file("/Users/keynes/Documents/Op.jpg")
#Op_face_encoding = face_recognition.face_encodings(Op_image)[0]
Morgane_image = face_recognition.load_image_file("/Users/keynes/Documents/Morgane.jpg")
Morgane_face_encoding = face_recognition.face_encodings(Morgane_image)[0]
#Andrew_image = face_recognition.load_image_file("/Users/keynes/Documents/Andrew.jpg")
#Andrew_face_encoding = face_recognition.face_encodings(Andrew_image)[0]

greeted = ["Unknown"]
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([Kaiyan_face_encoding], face_encoding)
        #match1 = face_recognition.compare_faces([Alex_face_encoding], face_encoding)
        #match2 = face_recognition.compare_faces([Julie_face_encoding], face_encoding)
        #match3 = face_recognition.compare_faces([Steve_face_encoding], face_encoding)
        #match4 = face_recognition.compare_faces([Chandan_face_encoding], face_encoding)
        #match8 = face_recognition.compare_faces([Rama_face_encoding], face_encoding)
        #match9 = face_recognition.compare_faces([Ram_face_encoding], face_encoding)
        #match5 = face_recognition.compare_faces([Op_face_encoding], face_encoding)
        match6 = face_recognition.compare_faces([Morgane_face_encoding], face_encoding)
        #match7 = face_recognition.compare_faces([Andrew_face_encoding], face_encoding)

        #name = "Unknown"
        if match[0]:
            name = "Ding"
        #if match3[0]:
            #name = "Steve"
        #if match4[0]:
            #name = "Chandan"
        #if match1[0]:
            #name = "Alex"
        #if match2[0]:
            #name = "Julie"
        #if match5[0]:
            #name = "Op"
        if match6[0]:
            name = "Morgane"
        #if match7[0]:
            #name = "Andrew"
        #if match8[0]:
            #name = "Rama"
        #if match9[0]:
            #name = "Ram"


        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        ### Allow program to speak
        if not name in greeted:
            os.system("say welcome" + name)
            print(name)
            greeted.append(name)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()