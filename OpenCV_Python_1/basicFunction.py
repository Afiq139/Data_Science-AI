import cv2
import numpy as np

kernel = np.ones((5,5), np.uint8)

print(kernel)

path = "Resources/L298N.png"
img = cv2.imread(path)
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(imgBlur, 100, 200)
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

cv2.imshow("L298N", img)
cv2.imshow("GrayScale", imgGray)
cv2.imshow("Img Blur", imgBlur)
cv2.imshow("Img Canny", imgCanny)
cv2.imshow("Img Dilation", imgDilation)
cv2.imshow("img Erosion", imgEroded)
cv2.waitKey(0)



