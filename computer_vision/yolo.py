import numpy as np
import cv2

net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

labels = open('coco.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layerNames)
    print(layerOutputs)

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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indices)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[classIds[i]]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = '{}: {:.2f}'.format(labels[classIds[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Yolo', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
