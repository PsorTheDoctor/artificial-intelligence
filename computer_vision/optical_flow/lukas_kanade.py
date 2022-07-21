import numpy as np
import cv2
from imutils.video import FileVideoStream
import time

vs = FileVideoStream('static/rec07_05-12_45.avi').start()

# ShiTomasi corner detection params
featureParams = dict(maxCorners = 100,
                     qualityLevel = 0.3,
                     minDistance = 7,
                     blockSize = 7)

# Lukas-Kanade optical flow params
lkParams = dict(winSize = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

oldFrame = vs.read()
oldFrame = cv2.resize(oldFrame, (0, 0), fx=0.6, fy=0.6)
oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(oldGray, mask=None, **featureParams)

# A mask for drawing purposes
mask = np.zeros_like(oldFrame)

while True:
    start = time.time()
    frame = vs.read()
    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, gray, p0, None, **lkParams)

    if p1 is not None:
        goodNew = p1[st == 1]
        goodOld = p0[st == 1]

    for i, (new, old) in enumerate(zip(goodNew, goodOld)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(img, 'FPS: {:.2f}'.format(fps), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('Lukas-Kanade', img)
    if cv2.waitKey(10) == 27:
        break

    # Update the previous frame and the previous points
    oldGray = gray.copy()
    p0 = goodNew.reshape(-1, 1, 2)

vs.stop()
cv2.destroyAllWindows()
