import numpy as np
import cv2
import time


def NMSBoxesFast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    # If boxes are integers convert them to floats
    if boxes.dtype.kind == 'i':
        boxes.astype('float')

    picked = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)
    # Sort the boxes by the bottom-right y coord
    indices = np.argsort(y2)

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        picked.append(i)

        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / area[indices[:last]]

        indices = np.delete(indices, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[picked].astype('int')


def NMSBoxesSlow(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    picked = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)
    # Sort the boxes by the bottom-right y coord
    indices = np.argsort(y2)

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        picked.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = indices[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        indices = np.delete(indices, suppress)

    return boxes[picked]


path = 'sunflower.jpg'
bboxes = np.array([
    (12, 30, 76, 94),
    (12, 36, 76, 100),
    (72, 36, 200, 164),
    (84, 48, 212, 176)
])

img = cv2.imread(path)
orig = img.copy()

for (startX, startY, endX, endY) in bboxes:
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

# Perform nms 1000 times to measure time
start = time.time()
picked = []
for _ in range(1000):
    picked = NMSBoxesFast(bboxes, 0.3)

end = time.time()
print(end - start)

start = time.time()
picked = []
for _ in range(1000):
    picked = NMSBoxesSlow(bboxes, 0.3)

end = time.time()
print(end - start)

for (startX, startY, endX, endY) in picked:
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow('Original', orig)
cv2.imshow('After NMS', img)
cv2.waitKey(0)
