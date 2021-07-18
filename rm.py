import numpy as np
import cv2
import copy

kernel = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture()
cap.open('red.avi')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    subt = cv2.subtract(img[:, :, 2], img[:, :, 0])
    thresh, ret = cv2.threshold(subt, 95, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(ret, kernel, iterations=1)
    prepro = cv2.erode(dilation, kernel, iterations=1)
    flood = copy.deepcopy(prepro)
    h, w = prepro.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(flood, mask, (0, 0), (255, 255, 255), flags=cv2.FLOODFILL_FIXED_RANGE)

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
                cropImg = flood[top_point_y:bottom_point_y, left_point_x:right_point_x]

                src = np.float32([box[0], box[1], box[2], box[3]])
                dst = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
                trans = cv2.getPerspectiveTransform(src, dst)
                afterTrans = cv2.warpPerspective(flood, trans, (int(width), int(height)))

                whitePx = np.count_nonzero(afterTrans)
                blackPx = area - whitePx
                ratio = blackPx / whitePx

                if ratio > 0.8:
                    cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
                elif ratio < 0.3:
                    rec = cv2.minAreaRect(contours[hire[0][i][2]])
                    boxx = cv2.boxPoints(rec)
                    boxx = np.int0(boxx)
                    wid = rec[1][0]
                    hei = rec[1][1]
                    rat = wid / height
                    print(rat)
                    if 0.25 < rat < 0.65:
                        center = (int(rec[0][0]), int(rec[0][1]))
                        print(center)
                        cv2.circle(img, center, 3, (0, 255, 0), -1)

                    cv2.drawContours(img, [boxx], 0, (0, 255, 0), 2)
                '''
                    bla = cv2.boundingRect(afterTrans)
                    center = (int(bla[0]+0.5*bla[2]), int(bla[1]+0.5*bla[3]))
                    cv2.circle(img, center, 3, (0, 255, 0), -1)
                    con, hir = cv2.findContours(flood, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    if len(con) > 0:
                        for t, cntr in enumerate(con):
                            rec = cv2.minAreaRect(cntr)
                            
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                    '''

    cv2.imshow('test', img)
    cv2.waitKey(3)
cv2.destroyAllWindows()
