import cv2
import numpy as np

img = cv2.imread('rice.jpg')
img = cv2.resize(img, (500, 500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

analysis = cv2.connectedComponentsWithStats(thresh, 10, cv2.CV_32S)
labels, labelIds, values, centroid = analysis
output = np.zeros(gray.shape, dtype=np.uint8)

print('Num of rice grains: {}'.format(labels))

for i in range(1, labels):
    newImg = img.copy()

    x = values[i, cv2.CC_STAT_LEFT]
    y = values[i, cv2.CC_STAT_TOP]
    w = values[i, cv2.CC_STAT_WIDTH]
    h = values[i, cv2.CC_STAT_HEIGHT]

    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cx, cy = centroid[i]

    cv2.rectangle(newImg, pt1, pt2, (0, 255, 0), 2)
    cv2.circle(newImg, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    component = np.zeros(gray.shape, dtype=np.uint8)
    mask = (labelIds == i).astype(np.uint8) * 255
    component = cv2.bitwise_or(component, mask)
    output = cv2.bitwise_or(output, mask)

    cv2.imshow('', newImg)
    cv2.imshow('Individual Component', component)
    cv2.imshow('Filtered Components', output)
    cv2.waitKey(0)
