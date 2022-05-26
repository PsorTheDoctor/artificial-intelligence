import numpy as np
import cv2
import time


def customNMSBoxes(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    # If boxes are integers convert them to floats
    if boxes.dtype.kind == 'i':
        boxes.astype('float')

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)
    # Sort the boxes by the bottom-right y coord
    idxs = np.argsort(y2)

    indices = []
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        indices.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return np.array(indices)


net = cv2.dnn.readNetFromDarknet('yolov2.cfg', 'yolov2.weights')

layerNames = net.getLayerNames()
layerNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

labels = open('coco.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    start = time.time()
    _, frame = cap.read()
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layerNames)

    boxes = []
    prevBoxes = []
    confidences = []
    prevConfidences = []
    classIds = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > 0.0:
                box = detection[0:4] * np.array([w, h, w, h])
                centerX, centerY, width, height = box.astype('int')
                x = int(centerX - width // 2)
                y = int(centerY - height // 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classId)

    # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, 0.0)
    # indices = customNMSBoxes(boxes, 1.0)
    indices = np.arange(len(boxes))
    print(indices)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = '{}: {:.2f}'.format(labels[classIds[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    prevBoxes = boxes
    prevConfidences = confidences

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(frame, 'FPS: {}'.format(round(fps, 2)), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('Yolo', frame)
    if cv2.waitKey(50) == 27:
        break

cap.release()
cv2.destroyAllWindows()
