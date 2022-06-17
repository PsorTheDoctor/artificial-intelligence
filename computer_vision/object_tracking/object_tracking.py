import cv2
import imutils

(major, minor) = cv2.__version__.split('.')[:2]

if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create('kfc'.upper())
else:
    trackers = {
        'csrt': cv2.TrackerCSRT_create(),
         #'kfc': cv2.TrackerKFC_create(),
        'boosting': cv2.TrackerBoosting_create(),
        'mil': cv2.TrackerMIL_create(),
        'tld': cv2.TrackerTLD_create(),
        'medianflow': cv2.TrackerMedianFlow_create(),
        'mosse': cv2.TrackerMOSSE_create(),
    }
    tracker = trackers['csrt']

windowId = 'Object tracking'
initBB = None
vs = cv2.VideoCapture(0)

while True:
    _, frame = vs.read()
    frame = imutils.resize(frame, width=500)
    (h, w) = frame.shape[:2]

    if initBB is not None:
        _, box = tracker.update(frame)

        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow(windowId, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        initBB = cv2.selectROI(windowId, frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
    elif key == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
