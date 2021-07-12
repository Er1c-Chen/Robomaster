import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier


winname = 'hi'
winname1 = 'hii'
kernel = np.ones((5,5),np.uint8)

cv2.namedWindow(winname, 0)
cv2.namedWindow(winname1, 0)

img = cv2.imread('img.jpg')
cap = cv2.VideoCapture()
cap.open('red.avi')

# clf = KNeighborsClassifier()

class distanceClassifier():
    def cls(self, x):
        cv2.imread('true.jpg')

        np.ndarray.flatten(x)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    subt = cv2.subtract(img[:, :, 2], img[:, :, 0])
    thresh, ret = cv2.threshold(subt, 95, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(ret, kernel, iterations=1)
    prepro = cv2.erode(dilation, kernel, iterations=1)
    '''
    h, w = prepro.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(prepro, mask, (0, 0), (255, 255, 255), flags=cv2.FLOODFILL_FIXED_RANGE)
    '''
    # cv2.imshow(winname1, prepro)

    contours, hire = cv2.findContours(prepro, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        for i, cnt in enumerate(contours):
            rect = cv2.minAreaRect(cnt)
            width = rect[1][0]
            height = rect[1][1]
            area = height * width
            if 8000 < area < 12000:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                left_point_x = int(np.min(box[:, 0]))
                right_point_x = int(np.max(box[:, 0]))
                top_point_y = int(np.min(box[:, 1]))
                bottom_point_y = int(np.max(box[:, 1]))
                cropImg = prepro[top_point_y:bottom_point_y, left_point_x:right_point_x]

                src = np.float32([box[0], box[1], box[2], box[3]])
                dst = np.float32([[0, height], [0, 0], [width, 0],  [width, height]])
                trans = cv2.getPerspectiveTransform(src, dst)
                afterTrans = cv2.warpPerspective(prepro, trans, (int(width), int(height)))

                if height < width:
                    height, width = width, height
                    trans_img = cv2.transpose(afterTrans)
                    target = cv2.flip(trans_img, 1)
                # label =
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)


    cv2.imshow(winname, img)
    # cv2.imshow(winname1, ret)
    cv2.waitKey(3)
cv2.destroyAllWindows()
