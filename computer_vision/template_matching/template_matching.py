import numpy as np
import cv2

cap = cv2.VideoCapture('przejazd_2.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

obj1 = cv2.resize(cv2.imread('object1.jpg', 0), (0, 0), fx=0.5, fy=0.5)
obj2 = cv2.resize(cv2.imread('object2.jpg', 0), (0, 0), fx=0.5, fy=0.5)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
res = cv2.VideoWriter('result.avi', fourcc, 24, (width // 2, height // 2))

def detect(img, obj, threshold=0.5, color=(0, 0, 255)):
    res = cv2.matchTemplate(img, obj, cv2.TM_CCOEFF_NORMED)
    h, w = obj.shape

    y, x = np.where(res >= threshold)
    if len(x) > 0 and len(y) > 0:
        cv2.rectangle(frame, (x[0], y[0]), (x[0] + w, y[0] + h), color, 5)

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    img = frame[:, :, 0]
    detect(img, obj1, threshold=0.6, color=(0, 0, 255))
    detect(img, obj2, threshold=0.6, color=(255, 0, 0))

    res.write(frame)
    cv2.imshow('Template matching', frame)
    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()
