import numpy as np
import cv2
from imutils.video import VideoStream

points = []
cropping = False
cropped = False


def crop(event, x, y, flags, param):
    global points, cropping, cropped

    # If left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        cropping = True

    # If left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        points.clear()
        cropping = False
        cropped = True

    # If clicked and not released yet
    elif cropping:
        if len(points) == 1:
            points.append((x, y))
        elif len(points) == 2:
            points[1] = (x, y)


windowId = 'Follow mode'
cv2.namedWindow(windowId)

vs = VideoStream(0).start()
frame = vs.read()
copy = frame.copy()
target = None

# Cropping loop
while not cropped:
    if not cropping:
        frame = vs.read()

    # Listening to the mouse events
    cv2.setMouseCallback(windowId, crop)

    copy = frame.copy()
    if len(points) == 2:
        target = copy[points[0][1]:points[1][1], points[0][0]:points[1][0]]
        cv2.rectangle(copy, points[0], points[1], (0, 255, 255), 2)

    cv2.imshow(windowId, copy)
    if cv2.waitKey(50) == 27:
        break

target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()  # Instead of xfeatures2d.SIFT_create()
kp, desc = sift.detectAndCompute(target, None)

indexParams = dict(algorithm=0, trees=5)
searchParams = dict()

flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# Tracking loop
while True:
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kpGray, descGray = sift.detectAndCompute(gray, None)
    matches = flann.knnMatch(desc, descGray, k=2)

    goodPoints = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            goodPoints.append(m)

    minMatchCount = 4
    if len(goodPoints) > minMatchCount:
        queryPts = np.float32([kp[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
        trainPts = np.float32([kpGray[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(queryPts, trainPts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()

        h, w = target.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)
    else:
        print('Not enough matches are found: {}/{}'.format(len(goodPoints), minMatchCount))

    cv2.imshow(windowId, frame)
    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()
vs.stop()
