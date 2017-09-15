"""
Created on Thu Mar 23 09:20:34 2017 by Kaiyan Ding
Updated on 07/20/2017 by Roald Gomes
Updated on 07/20/2017 by Morgane Della Valle

"""

import glob
import datetime

import boto3
import cv2
import face_recognition

# os.sched_setaffinity(0, {0, 1, 2, 3})

__author__ = 'keynes'
__version__ = "3.0"
__maintainer__ = "Roald Gomes"
__email__ = "roald.gomes-filho@capgemini.com"

# This is a super simple demo of running face recognition on live video from your webcam.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Default Resolution
camera_frame_width = 800
camera_frame_height = 480

# Set resolution to FHD (1920 x 1080)
# camera_frame_width = 1920
# camera_frame_height = 1080
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_frame_width)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_frame_height)


#################### Setting up parameters ################
frames_skip = 4
images_path = '/home/amundsen/PycharmProjects/walmart'
SmsTime = 60
SmsTopicArn = 'arn:aws:sns:us-east-1:750717214473:facial_recognition'
SmsMessage = 'Unknown person detected on camera #1'

###########################################################

# Instantiate sns Client
sns = boto3.client('sns')
SmsDateTime = datetime.datetime.now() - datetime.timedelta(seconds=SmsTime)

# Load pictures from path and learn how to recognize them
picture_files = glob.glob(images_path + '/*.jpg')

person_preffixes = [j.split('/')[-1] for j in [i.split('@')[0] for i in picture_files]]
person_preffixes.append('')

person_names = [j.split('.')[0] for j in [i.split('@')[1] for i in picture_files]]
person_names.append('Unknown')

person_images = [face_recognition.load_image_file(i) for i in picture_files]
person_face_encodings = [face_recognition.face_encodings(i)[0] for i in person_images]

greeted = []
while True:
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        person_matches = [face_recognition.compare_faces([i], face_encoding) for i in person_face_encodings]

        try:
            person_index = person_matches.index([True])

        except ValueError:
            person_index = person_names.index('Unknown')

            # If SMS was sent more than 60s ago - Send SMS
            if int((datetime.datetime.now() - SmsDateTime).total_seconds()) >= 60:
                sns.publish(TopicArn=SmsTopicArn, Message=SmsMessage)
                SmsDateTime = datetime.datetime.now()

        if person_names[person_index] == 'Unknown':
            # Draw a thick red box around the face if the person is unknown
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)

            # Draw a red label with a name below the face
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, 'Unknown', (left + 2, bottom - 2), font, 1, (255, 255, 255), 1)

            # Display an alert when second is even to have the flashing effect
            if datetime.datetime.now().second % 2 == 0:
                rec_top = 20
                rec_left = int(camera_frame_width / 2) - 200
                rec_bottom = 40
                rec_right = int(camera_frame_width / 2) + 200
                cv2.rectangle(frame, (rec_left, rec_top), (rec_right, rec_bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(frame, 'UNKNOWN PERSON DETECTED!', (rec_left + 20, rec_bottom - 4), font, 1, (0, 255, 255), 1)

        else:
            # Draw a green box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (51, 255, 51), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (51, 255, 51), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, person_preffixes[person_index] + '. ' + person_names[person_index],
                        (left + 2, bottom - 2), font, 1, (0, 0, 0), 1)

        print(person_names[person_index])
        greeted.append((person_names[person_index], datetime.datetime.now().strftime('%c')))

    # Display the resulting image
    cv2.imshow('Facial Recognition', frame)

    # Skip frames
    for count in range(frames_skip):
        ret1, frame1 = video_capture.read()

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

# Write greeted to a file
file_name = images_path + '/detected_people.csv'
f = open(file_name, 'w')
[f.write(','.join(i) + '\n') for i in greeted]
f.close()

# Print greeted to console
[print(i) for i in greeted]

# greeted.to_csv("detected_people.csv")
