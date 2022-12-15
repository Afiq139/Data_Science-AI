import cv2

img = cv2.imread("Resources/L298N.png")

cv2.imshow("L298N", img)

# delay infinity = 0
# delay 1000 = 1000 ms
cv2.waitKey(0)