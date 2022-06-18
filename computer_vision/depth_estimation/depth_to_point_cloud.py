import numpy as np
import cv2
from imutils.video import VideoStream
import matplotlib.pyplot as plt

Q = np.array([[1, 0, 0, -160],
              [0, 1, 0, -120],
              [0, 0, 0, 350],
              [0, 0, 1/90, 0]], dtype=np.float32)

net = cv2.dnn.readNet('model-small.onnx')

vs = VideoStream(0).start()
(h, w) = vs.read().shape[:2]
frame = vs.read()

# Mean subtraction to prevent illumination changes
mean = np.mean(frame, axis=(0, 1))
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), mean, swapRB=True, crop=False)
net.setInput(blob)

depthMap = net.forward()
depthMap = depthMap[0, :, :]
depthMap = cv2.resize(depthMap, (32, 32))
depthMap = cv2.normalize(depthMap, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

points = cv2.reprojectImageTo3D(depthMap, Q, handleMissingValues=False)
maskMap = depthMap > 0.4
output_points = points[maskMap]
x = output_points[:, 0]
y = output_points[:, 1]
z = output_points[:, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(x, y, z, c=z, cmap='gray')
ax.view_init(45, 45)
plt.show()
