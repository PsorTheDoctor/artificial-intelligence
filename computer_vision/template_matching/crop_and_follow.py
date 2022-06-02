import numpy as np
import cv2
from imutils.video import FileVideoStream

points = []
cropping = False


def crop(event, x, y, flags, param):
    global points, cropping

    # If left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        cropping = True

    # If left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        points.clear()
        cropping = False

    # If clicked and not released yet
    elif cropping:
        if len(points) == 1:
            points.append((x, y))
        elif len(points) == 2:
            points[1] = (x, y)


def follow(img, target, threshold=0.7):

    img = img[:, :, 0]
    target = target[:, :, 0]

    res = cv2.matchTemplate(img, target, cv2.TM_CCOEFF_NORMED)
    (h, w) = target.shape
    (y, x) = np.where(res >= threshold)

    if len(x) > 0 and len(y) > 0:
        return (x[0], y[0]), (x[0] + w, y[0] + h)

    return (None, None)


windowId = 'Follow mode'
cv2.namedWindow(windowId)

vs = FileVideoStream('example.mp4').start()
frame = vs.read()
copy = frame.copy()
target = None

while True:
    if not cropping:
        frame = vs.read()

    # Listening to mouse events
    cv2.setMouseCallback(windowId, crop)

    copy = frame.copy()
    if len(points) == 2:
        target = copy[points[0][1]:points[1][1], points[0][0]:points[1][0]]
        cv2.rectangle(copy, points[0], points[1], (0, 255, 255), 2)

    if not cropping and target is not None:
        (topLeft, bottomRight) = follow(copy, target)
        if topLeft and bottomRight:
            cv2.rectangle(copy, topLeft, bottomRight, (0, 255, 0), 2)

    cv2.imshow(windowId, copy)
    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()
vs.stop()
