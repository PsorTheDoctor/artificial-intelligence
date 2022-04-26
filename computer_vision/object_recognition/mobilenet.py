import numpy as np
import cv2
from imutils.video import VideoStream
import time

dims = (300, 300)
normRatio = 0.007843

net = cv2.dnn.readNetFromCaffe('../data/mobilenet/mobilenet.prototxt.txt',
                               '../data/mobilenet/mobilenet.caffemodel')
classes = open('../data/mobilenet/classes.txt').read().split('\n')

colors = np.random.uniform(255, 0, size=(len(classes), 3))

vs = VideoStream(0).start()
# cap = cv2.VideoCapture(0)

while True:
    start = time.time()
    # _, frame = cap.read()
    frame = vs.read()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, dims), normRatio, dims, 127.5)
    net.setInput(blob)
    output = net.forward()

    for i in np.arange(0, output.shape[2]):
        confidence = output[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(output[0, 0, i, 1])
            box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype('int')

            label = '{}: {:.2f}'.format(classes[idx], confidence)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15

            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colors[idx], 2)

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(frame, 'FPS: {}'.format(fps), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow('MobileNet', frame)

    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()
vs.stop()
