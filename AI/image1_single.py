import cv2
from random import randrange
#load pre-trained data

face_trained_data = cv2.CascadeClassifier('C:/Users/shafi/Desktop/Data_Science-AI/AI/data/haarcascade_frontalface_alt.xml') 

img = cv2.imread('C:/Users/shafi/Desktop/Data_Science-AI/AI/images/joey.jpg')
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_trained_data.detectMultiScale(grayscale_img)


for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3)

#Display the image in windows   
cv2.imshow('color', img)
cv2.imshow("B&W",grayscale_img)
cv2.waitKey()