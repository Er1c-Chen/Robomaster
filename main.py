import cv2
import numpy as np

img = cv2.imread('i.jpg', 0)
img1 = cv2.imread('i.jpg')
contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
cv2.namedWindow('hi')
print(len(contours))
for i, contour in enumerate(contours):
    cv2.drawContours(img, contours, i, (0, 0, 255), 2)

if len(contours) > 0:
    cnt = contours[0]
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    (x, y, radius) = np.int0((x, y, radius))
    cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
    print('x:', x, 'y:', y)

cv2.imshow('hi', img)
cv2.waitKey(0)
cv2.destroyWindow('hi')