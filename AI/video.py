import cv2
from random import randrange
#load pre-trained data

face_trained_data = cv2.CascadeClassifier('data\haarcascade_frontalface_alt.xml') 

#load webcam
webcam = cv2.VideoCapture(0)

while True:
    boolean_frame_read_successful, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_trained_data.detectMultiScale(grayscale_img)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow('friends', frame)

    key = cv2.waitKey(5)

    if key == 81 or key == 113: #if click q or Q
        break