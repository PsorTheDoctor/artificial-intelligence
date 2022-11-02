import numpy as np
import cv2
import imutils

img = cv2.imread('example.jpg', cv2.COLOR_BGR2GRAY)
template = cv2.imread('template.jpg', cv2.COLOR_BGR2GRAY)
threshold = 0.5
found = None

for scale in np.linspace(0.1, 10, 100)[::-1]:
    resized = imutils.resize(template, width=int(template.shape[1] * scale))
    ratio = template.shape[1] / resized.shape[1]
    th, tw = resized.shape[:2]

    if resized.shape[0] < img.shape[0] and resized.shape[0] < img.shape[1]:
        res = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(res)

        if maxVal > 0.5:
            print(maxVal)
            clone = img.copy()
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0]+tw, maxLoc[1]+th), (0, 255, 0), 2)
            cv2.imshow('Multi-scale matching', clone)
            cv2.waitKey(0)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, ratio)
        else:
            print('Nothing has been found.')

_, maxLoc, ratio = found
startX, startY = (maxLoc[0]*ratio, maxLoc[1]*ratio)
endX, endY = ((maxLoc[0]+tw)*ratio, (maxLoc[1]+th)*ratio)

cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow('', img)
cv2.waitKey(0)
