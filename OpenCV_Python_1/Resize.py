import cv2

path = "Resources/L298N.png"
img = cv2.imread(path)
print(img.shape)

width , height = 1000 , 1000
imgResize = cv2.resize(img, (width, height))
print(imgResize.shape)

cv2.imshow("Road", img)
cv2.imshow("Road Resized", imgResize)

cv2.waitKey(0)