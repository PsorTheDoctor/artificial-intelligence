import numpy as np
import cv2

cap = cv2.VideoCapture('static/rec07_05-12_52.avi')

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 300,
                            minLineLength=50,
                            maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow('Hough line transform', frame)
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
