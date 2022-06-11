import numpy as np
import cv2
from sort import *

tracker = Sort()
memory = {}

net = cv2.dnn.readNetFromDarknet('yolov2.cfg', 'yolov2.weights')

layerNames = net.getLayerNames()
layerNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

labels = open('coco.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

cap = cv2.VideoCapture('example2.mp4')

while True:
    (_, frame) = cap.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layerNames)

    boxes = []
    confidences = []
    classIds = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                centerX, centerY, width, height = box.astype('int')
                x = int(centerX - width // 2)
                y = int(centerY - height // 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classId)

    indices = np.arange(len(boxes))
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    dets = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIds = []
    prev = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIds.append(int(track[4]))
        memory[indexIds[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = 0
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in colors[indexIds[i] % len(colors)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIds[i] in prev:
                prevBox = prev[indexIds[i]]
                (x2, y2) = (int(prevBox[0]), int(prevBox[1]))
                (w2, h2) = (int(prevBox[2]), int(prevBox[3]))

            # text = '{}'.format(indexIds[i])
            # cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    cv2.imshow('Yolo + Sort', frame)
    if cv2.waitKey(50) == 27:
        break

cap.release()
cv2.destroyAllWindows()
