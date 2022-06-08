import numpy as np
import cv2
from imutils.video import VideoStream

net = cv2.dnn.readNet('model-small.onnx')

vs = VideoStream(0).start()
(h, w) = vs.read().shape[:2]

while True:
    frame = vs.read()

    # Mean subtraction to prevent illumination changes
    mean = np.mean(frame, axis=(0, 1))
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), mean, swapRB=True, crop=False)
    net.setInput(blob)

    depthMap = net.forward()
    depthMap = depthMap[0, :, :]
    depthMap = cv2.resize(depthMap, (w, h))
    depthMap = cv2.normalize(depthMap, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    mask = np.where(depthMap > 0.5, 1, 0).astype(np.uint8)

    cv2.imshow('3D Projection', frame * mask[:, :, np.newaxis])
    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()
vs.stop()
