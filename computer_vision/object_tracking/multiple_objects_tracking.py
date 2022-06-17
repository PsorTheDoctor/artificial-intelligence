import cv2
import imutils

trackers = {
    'csrt': cv2.TrackerCSRT_create(),
    #'kfc': cv2.TrackerKFC_create(),
    'boosting': cv2.TrackerBoosting_create(),
    'mil': cv2.TrackerMIL_create(),
    'tld': cv2.TrackerTLD_create(),
    'medianflow': cv2.TrackerMedianFlow_create(),
    'mosse': cv2.TrackerMOSSE_create(),
}
# MultiTracker doesn't work with tested opencv versions!
multiTracker = cv2.MultiTracker_create()

windowId = 'Object tracking'
vs = cv2.VideoCapture(0)

while True:
    _, frame = vs.read()
    frame = imutils.resize(frame, width=600)
    _, boxes = multiTracker.update(frame)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow(windowId, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        box = cv2.selectROI(windowId, frame, fromCenter=False, showCrosshair=True)
        tracker = trackers['csrt']
        multiTracker.add(tracker, frame, box)
    elif key == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
