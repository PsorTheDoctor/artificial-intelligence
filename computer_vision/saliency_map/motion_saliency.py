from imutils.video import VideoStream
import cv2

saliency = None
vs = VideoStream(0).start()

while True:
    frame = vs.read()
    h, w = frame.shape[:2]

    if saliency is None:
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(w, h)
        saliency.init()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (success, saliencyMap) = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype('uint8')

    cv2.imshow('Frame', frame)
    cv2.imshow('Saliency', saliencyMap)
    if cv2.waitKey(10) == 27:
        break

vs.stop()
cv2.destroyAllWindows()
