import cv2

path = "Resources/L298N.png"
img = cv2.imread(path)
print(img.shape)

width , height = 1000 , 1000
imgResize = cv2.resize(img, (width, height))
print(imgResize.shape)

imgCropped = img[200:540, 0:900]
# img.shape[1] = width, img.shape[0] = height
imgCropResize = cv2.resize(imgCropped, (img.shape[1], img.shape[0]))

cv2.imshow("L298N", img)
# cv2.imshow("L298N Resized", imgResize)
cv2.imshow("L298N Cropped", imgCropped)
cv2.imshow("L298N Cropped Resized", imgCropResize)


cv2.waitKey(0)