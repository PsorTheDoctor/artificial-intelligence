import numpy as np
import cv2
from imutils.video import VideoStream
import time

# Model can be downloaded from:
# https://github.com/doleron/yolov5-opencv-cpp-python/tree/main/config_files
net = cv2.dnn.readNet('data/yolov5s.onnx')

labels = open('data/coco.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

vs = VideoStream(0).start()

while True:
    start = time.time()
    frame = vs.read()
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    preds = net.forward()

    boxes = []
    confidences = []
    classIds = []

    output = preds[0]
    x_factor = w / 640
    y_factor = h / 640

    for i in range(25200):
        row = output[i]
        confidence = row[4]

        if confidence > 0.5:
            scores = row[5:]
            _, _, _, maxIdx = cv2.minMaxLoc(scores)
            classId = maxIdx[1]

            if scores[classId] > 0.25:
                confidences.append(confidence)
                classIds.append(classId)

                (x, y) = (row[0].item(), row[1].item())
                (w, h) = (row[2].item(), row[3].item())

                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[classIds[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.2f}'.format(labels[classIds[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(frame, 'FPS: {}'.format(round(fps, 2)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('Yolo v5', frame)
    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()
vs.stop()
